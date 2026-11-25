"""topoflow collect | sources | show | review | decide"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import grid


def _cmd_collect(a) -> int:
    from .gatekeeper import DiskTooFull
    from .orchestrator import collect

    req = grid.make_request(a.region, a.resolution, a.start, a.end,
                            [int(h) for h in a.hours.split(",")] if a.hours else None,
                            tuple(float(x) for x in a.bbox.split(",")) if a.bbox else None)
    try:
        run = collect(req, a.mode, out_root=a.out, fixtures_dir=a.fixtures, cache_dir=a.cache,
                      skip=set(a.skip.split(",")) if a.skip else None)
    except DiskTooFull as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return _show(run)


def _show(run: Path) -> int:
    m = json.loads((Path(run) / "manifest.json").read_text())
    print(f"\nrun {m['run_id']}  ({m['mode']})")
    print("\nsources")
    for s in m["source_runs"]:
        extra = f"  -> {s['person_route']}" if s["status"] in ("barred", "missing") and s["person_route"] else ""
        print(f"  {s['status']:8} {s['source_id']:16} {s['calls']:4} calls {s['bytes'] / 1e6:7.2f} MB  {s['reason']}{extra}")
    print("\nfields")
    for k, f in m["fields"].items():
        c = f" [{f['conflict'].get('verdict', 'CONFLICT')} vs {f['conflict']['against']}, rel rms {f['conflict']['relative_rms']}]" if f.get("conflict") else ""
        print(f"  {f['state']:11} {k:26} {f['selected'] or '':16}{c}")
    print("\nrefused proposals")
    n = 0
    for r in m["agent_runs"]:
        for rej in r["rejected"]:
            n += 1
            print(f"  {r['agent_id']:9} {rej['field']:24} {','.join(rej['codes'])}: {'; '.join(rej['reasons'])[:160]}")
    if not n:
        print("  none")
    print("\ncross-checks")
    for c in m["cross_checks"]:
        print(f"  {c['status']:7} {c['name']}\n          {c['detail']}")
    for label in m["labels"]:
        print(f"\n! {label}")
    return 0


def _cmd_show(a) -> int:
    from .store import latest_run

    run = Path(a.run) if a.run else latest_run(a.out)
    if not run:
        print("no runs yet; run `python -m topoflow collect` first", file=sys.stderr)
        return 1
    return _show(run)


def _cmd_sources(_a) -> int:
    from .sources.registry import SOURCES

    for s in SOURCES.values():
        print(f"{s.id:16} tier {s.tier}  {s.kind:16} {s.name}")
        print(f"{'':16} licence: {s.license}")
        print(f"{'':16} terms checked: {s.terms_checked}  {s.terms_url}")
        if s.needs:
            print(f"{'':16} needs: {s.needs}")
        print()
    return 0


def _cmd_review(a) -> int:
    from .review import run_review
    from .store import latest_run

    run = Path(a.run) if a.run else latest_run(a.out)
    if not run:
        print("no runs yet", file=sys.stderr)
        return 1
    summary = run_review(run, effort=a.effort)
    print(summary)
    return 0


def _cmd_decide(a) -> int:
    from .review import decide, list_proposals
    from .store import latest_run

    run = Path(a.run) if a.run else latest_run(a.out)
    if a.proposal_id is None:
        for p in list_proposals(run):
            state = p.get("decision", {}).get("decision", "pending")
            print(f"{p['id']}  {state:8} {p['type']:9} {p.get('field') or p.get('check') or ''}: {p['text'][:120]}")
        return 0
    decide(run, a.proposal_id, a.decision, a.by, a.reason)
    print(f"recorded: {a.proposal_id} {a.decision} by {a.by}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="topoflow", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="gather every source for a region and window, gate it, derive, and store a run")
    c.add_argument("--region", default="permian", help=f"one of {', '.join(grid.REGIONS)}")
    c.add_argument("--bbox", help="west,south,east,north (overrides the region's box)")
    c.add_argument("--resolution", type=float, help="grid step in degrees (default depends on region)")
    c.add_argument("--start", default=grid.DEFAULT_START)
    c.add_argument("--end", default=grid.DEFAULT_END)
    c.add_argument("--hours", help="UTC hours, comma-separated (default 0,6,12,18)")
    c.add_argument("--mode", choices=["live", "record", "replay"], default="live")
    c.add_argument("--skip", help="source ids to skip, comma-separated")
    c.add_argument("--out", default="data/runs")
    c.add_argument("--fixtures", default="fixtures")
    c.add_argument("--cache", default="data/cache")
    c.set_defaults(fn=_cmd_collect)

    s = sub.add_parser("show", help="print a run's sources, fields, refusals and cross-checks")
    s.add_argument("run", nargs="?")
    s.add_argument("--out", default="data/runs")
    s.set_defaults(fn=_cmd_show)

    sub.add_parser("sources", help="list every source with its licence and what it needs").set_defaults(fn=_cmd_sources)

    r = sub.add_parser("review", help="ask the Claude review agent to read a run and propose flags (needs ANTHROPIC_API_KEY)")
    r.add_argument("run", nargs="?")
    r.add_argument("--out", default="data/runs")
    r.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    r.set_defaults(fn=_cmd_review)

    d = sub.add_parser("decide", help="list review proposals, or accept/reject one")
    d.add_argument("proposal_id", nargs="?")
    d.add_argument("decision", nargs="?", choices=["accept", "reject"])
    d.add_argument("--by", default="", help="your name; decisions are attributed")
    d.add_argument("--reason", default="")
    d.add_argument("--run")
    d.add_argument("--out", default="data/runs")
    d.set_defaults(fn=_cmd_decide)

    a = ap.parse_args(argv)
    if a.cmd == "decide" and a.proposal_id and not (a.decision and a.by and a.reason):
        ap.error("decide <id> accept|reject needs --by and --reason")
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
