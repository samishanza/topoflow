"""One way out to the network, in three modes.

live    fetch, nothing saved
record  reuse a recording when there is one, else fetch and save the response with where and when it came from,
        so an interrupted run resumes where it stopped
replay  never touch the network; a request with no recording fails, and the source reports itself missing

Recordings are real public data, labelled `dataOrigin: real`. They are keyed by URL and sorted params, so a replay
asks the same question a live run asked.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import requests

from .contract import now_iso

Mode = Literal["live", "record", "replay"]
USER_AGENT = "topoflow/0.2 (research prototype; https://github.com/)"


class NoRecording(RuntimeError):
    pass


class DeadlineExceeded(RuntimeError):
    pass


@dataclass
class CallLog:
    calls: int = 0
    bytes: int = 0
    ms: int = 0
    replayed: bool = False
    urls: list[str] = field(default_factory=list)


def _provider(host: str) -> str:
    """Rate budgets belong to a provider, not a hostname: Open-Meteo's archive and air-quality hosts share one."""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _key(url: str, params: dict[str, Any] | None) -> str:
    canon = json.dumps([url, sorted((params or {}).items())], default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


class Fetcher:
    def __init__(self, mode: Mode = "live", fixtures_dir: str | Path = "fixtures", deadline_s: float = 900,
                 session: requests.Session | None = None, no_record: set[str] | None = None,
                 cache_dir: str | Path = "data/cache"):
        self.mode = mode
        self.fixtures_dir = Path(fixtures_dir)
        self.cache_dir = Path(cache_dir)
        # Sources whose licence bars redistribution record into the gitignored cache, never into fixtures/.
        self.no_record = no_record or set()
        self.deadline = time.monotonic() + deadline_s
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.logs: dict[str, CallLog] = {}
        # Open-Meteo publishes its budget in headers; remember it so we wait instead of getting refused.
        self._rate: dict[str, tuple[float, float]] = {}  # provider -> (remaining, reset_epoch)

    def log(self, source_id: str) -> CallLog:
        return self.logs.setdefault(source_id, CallLog())

    def _dir_for(self, source_id: str) -> Path:
        base = self.cache_dir / "recordings" if source_id in self.no_record else self.fixtures_dir
        return base / source_id

    def _find_recording(self, source_id: str, key: str) -> Path | None:
        for base in (self.fixtures_dir / source_id, self.cache_dir / "recordings" / source_id):
            for suffix in (".json.gz", ".json"):
                p = base / f"{key}{suffix}"
                if p.exists():
                    return p
        return None

    def _wait_for_budget(self, provider: str, cost: float):
        if provider not in self._rate:
            return
        remaining, reset = self._rate[provider]
        if remaining >= cost:
            return
        wait = max(0.0, reset - time.time()) + 1
        if time.monotonic() + wait > self.deadline:
            raise DeadlineExceeded(f"{provider} rate budget resets in {wait:.0f}s, past the run deadline")
        time.sleep(wait)

    def get_json(self, source_id: str, url: str, params: dict[str, Any] | None = None, *, cost: float = 1.0,
                 timeout: float = 120) -> Any:
        return self._get(source_id, url, params, cost=cost, timeout=timeout, kind="json")

    def get_text(self, source_id: str, url: str, params: dict[str, Any] | None = None, timeout: float = 120) -> str:
        return self._get(source_id, url, params, cost=1.0, timeout=timeout, kind="text")

    def _get(self, source_id, url, params, *, cost, timeout, kind):
        log = self.log(source_id)
        key = _key(url, params)
        full = f"{url}?{urlencode(params, doseq=True)}" if params else url
        log.urls.append(full if len(full) < 300 else full[:297] + "...")
        t0 = time.monotonic()

        path = self._find_recording(source_id, key) if self.mode in ("replay", "record") else None
        if self.mode == "replay" and not path:
            raise NoRecording(f"no recorded response for {source_id} {url}")
        if path:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as fh:
                rec = json.load(fh)
            log.calls += 1
            log.replayed = True
            log.ms += int((time.monotonic() - t0) * 1000)
            return rec["response"]

        if time.monotonic() > self.deadline:
            raise DeadlineExceeded(f"run deadline passed before {source_id} could ask {url}")
        host = requests.utils.urlparse(url).hostname or ""
        provider = _provider(host)
        for attempt in range(4):
            self._wait_for_budget(provider, cost)
            resp = self.session.get(url, params=params, timeout=timeout)
            if "X-RateLimit-Remaining" in resp.headers:
                try:
                    self._rate[provider] = (float(resp.headers["X-RateLimit-Remaining"]),
                                            float(resp.headers["X-RateLimit-Reset"]))
                except (KeyError, ValueError):
                    pass
            log.calls += 1
            log.bytes += len(resp.content)
            if resp.status_code != 429:
                break
            # A per-minute limit clears in a minute: wait and retry. An hourly or daily one ends this source's run.
            if "minute" not in resp.text.lower() or attempt == 3 or time.monotonic() + 61 > self.deadline:
                raise RuntimeError(f"{host} refused with 429 Too Many Requests: {resp.text[:200]}")
            time.sleep(61)
        log.ms += int((time.monotonic() - t0) * 1000)
        resp.raise_for_status()
        body = resp.json() if kind == "json" else resp.text
        if kind == "json" and isinstance(body, dict) and body.get("error"):
            raise RuntimeError(f"{host} answered with an error: {body.get('reason') or body.get('error')}")

        if self.mode == "record":
            d = self._dir_for(source_id)
            d.mkdir(parents=True, exist_ok=True)
            rec = {
                "label": f"Real public data from {host}. Recorded response, not synthetic.",
                "dataOrigin": "real",
                "url": url,
                "params": params,
                "retrievedAt": now_iso(),
                "status": resp.status_code,
                "response": body,
            }
            text = json.dumps(rec, separators=(",", ":"), default=str)
            # Gzip anything big; the grids compress about tenfold.
            if len(text) > 50_000:
                with gzip.open(d / f"{key}.json.gz", "wt", encoding="utf-8") as fh:
                    fh.write(text)
            else:
                (d / f"{key}.json").write_text(text + "\n", encoding="utf-8")
        return body

    def download(self, source_id: str, url: str, dest: Path, timeout: float = 600) -> Path:
        """Bulk files go to the cache, never to fixtures. Replay uses the cached file or fails."""
        log = self.log(source_id)
        log.urls.append(url)
        if dest.exists():
            log.replayed = self.mode == "replay"
            return dest
        if self.mode == "replay":
            raise NoRecording(f"{dest} is not cached and replay mode cannot download it")
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        with self.session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    log.bytes += len(chunk)
            os.replace(tmp, dest)
        log.calls += 1
        log.ms += int((time.monotonic() - t0) * 1000)
        return dest
