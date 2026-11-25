from __future__ import annotations

from typing import Literal

from ..contract import now_iso
from ..http import Fetcher
from .registry import SOURCES


class SourceUnavailable(Exception):
    """A source that cannot answer for a stated reason: no key, outside its coverage, or not reachable."""

    def __init__(self, status: Literal["missing", "barred", "skipped", "error"], reason: str, person_route: str = ""):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.person_route = person_route


def doc_fields(fx: Fetcher, source_id: str, label_suffix: str = "") -> dict:
    """The SourceDoc fields every adapter fills the same way."""
    entry = SOURCES[source_id]
    log = fx.log(source_id)
    return dict(
        source_id=source_id,
        refs=list(dict.fromkeys(log.urls))[:20] or [entry.url],
        label=entry.name + (f" ({label_suffix})" if label_suffix else ""),
        retrieved_at=now_iso(),
        data_origin="real",
        replayed=log.replayed,
    )

