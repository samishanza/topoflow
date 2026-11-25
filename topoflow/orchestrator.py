"""One run: preflight, discovery agents cheapest-first behind the gate, reconcile, derive, cross-check, store.

There is no path from an agent to the record that skips gate(). That is the point of this module.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from . import crosschecks, grid
from .agents.base import AgentContext
from .agents.discovery import DISCOVERY_AGENTS
from .catalog import FIELDS
from .contract import AgentRunReport, FieldRecord, Rejection, RunManifest, RunRequest, now_iso
from .derive import derive, reconcile
from .gatekeeper import budgets_for, check_budget, check_disk, gate
from .http import Fetcher, Mode
from .sources import TOOLS, SourceClient
from .sources.registry import SOURCES
from .store import write_run


def run_id_for(req: RunRequest) -> str:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{req.region}-{req.resolution_deg:g}deg-{req.start}_{req.end}"


def collect(req: RunRequest, mode: Mode = "live", *, out_root: str | Path = "data/runs",
            fixtures_dir: str | Path = "fixtures", cache_dir: str | Path = "data/cache",
            skip: set[str] | None = None, agents=DISCOVERY_AGENTS,
            tools: dict[str, tuple[str, Callable[..., Any]]] | None = None,
            labels: list[str] | None = None, progress: Callable[[str], None] = print) -> Path:
    out_root = Path(out_root)
    free = check_disk(out_root, req)
    progress(f"preflight: {free:.0f} MB free; grid {len(grid.axes(req)[0])}x{len(grid.axes(req)[1])}, "
             f"{len(grid.times(req))} times, mode={mode}")

    fx = Fetcher(mode, fixtures_dir=fixtures_dir, cache_dir=cache_dir,
                 no_record={s.id for s in SOURCES.values() if s.no_redistribution})
    sources = SourceClient(fx, req, tools=tools or TOOLS, skip=skip)
    known: dict[str, Any] = {}
    untrusted: dict[str, str] = {}
    watchdog: list[str] = []
    accepted_all = []
    reports: list[AgentRunReport] = []
    budgets = budgets_for(req)

    for AgentCls in agents:
        agent = AgentCls()
        progress(f"agent {agent.id}: {agent.purpose}")
        before = {k: (v.calls, v.bytes) for k, v in fx.logs.items()}
        ctx = AgentContext(request=req, sources=sources, known=known, allowed_tools=agent.tools)
        t0 = time.monotonic()
        result = agent.run(ctx)
        wall = time.monotonic() - t0
        calls = sum(v.calls - before.get(k, (0, 0))[0] for k, v in fx.logs.items())
        nbytes = sum(v.bytes - before.get(k, (0, 0))[1] for k, v in fx.logs.items())
        out_of_scope = sorted({t for t in result.tool_calls if t not in agent.tools})
        for p in result.proposals:
            if p.evidence and p.evidence.untrusted_text:
                untrusted[p.evidence.source_id] = p.evidence.untrusted_text
        watchdog.extend(n for n in result.notes if n.startswith("instruction-like") and n not in watchdog)
        breach = check_budget(agent.id, budgets[agent.id], len(result.tool_calls), calls, nbytes, wall)
        if breach:
            g_acc, g_rej = [], [Rejection(agent_id=agent.id, field=p.field,
                                          source_id=p.evidence.source_id if p.evidence else None,
                                          codes=["budget_exceeded"],
                                          reasons=[f"{breach.limit}: used {breach.used:g}, allowed {breach.allowed:g}"])
                                for p in result.proposals]
        else:
            g = gate(result.proposals, req, known)
            g_acc, g_rej = g.accepted, g.rejected
        accepted_all.extend(g_acc)
        reports.append(AgentRunReport(
            agent_id=agent.id, tools_called=result.tool_calls, out_of_scope_tools=out_of_scope,
            accepted=[f"{a.proposal.field} <- {a.proposal.evidence.source_id}" for a in g_acc],
            rejected=g_rej, could_not_resolve=[k for k in result.could_not_resolve if k not in known],
            notes=result.notes, wall_ms=int(wall * 1000), budget_breach=breach,
        ))
        progress(f"  {len(g_acc)} admitted, {len(g_rej)} refused, {calls} HTTP calls, {nbytes / 1e6:.1f} MB, {wall:.0f}s")

    selected, records, by_field = reconcile(accepted_all)
    known = dict(selected)

    progress("derive: CH4 mole fraction, divergence, vorticity, critical points")
    ctx = AgentContext(request=req, sources=sources, known=known)
    t0 = time.monotonic()
    d = derive(req, known, ctx)
    g = gate(d.proposals, req, dict(known))
    reports.append(AgentRunReport(agent_id="derive", tools_called=[], out_of_scope_tools=[],
                                  accepted=[a.proposal.field for a in g.accepted], rejected=g.rejected,
                                  could_not_resolve=d.missed, notes=d.notes, wall_ms=int((time.monotonic() - t0) * 1000)))
    selected, records, by_field = reconcile(accepted_all + g.accepted)
    known = dict(selected)

    progress("cross-checks")
    checks = crosschecks.run_all(req, known)

    # Every catalog field the run could not fill is recorded as missing, with whoever tried.
    for key, spec in FIELDS.items():
        if key not in records:
            records[key] = FieldRecord(field=key, state="missing", unit=spec.unit,
                                       note="; ".join(f"{r.agent_id} could not resolve" for r in reports
                                                      if key in r.could_not_resolve) or "no source answered")

    used = {r.source_id for r in sources.runs.values() if r.status == "ok"}
    run_labels = list(labels or [])
    if any(c.data_origin == "synthetic" for rec in records.values() for c in rec.candidates):
        run_labels.append("CONTAINS SYNTHETIC DATA")
    run_labels.append("Research prototype. Derived fields (CH4 mole fraction, divergence, critical points) are inferred, not observed.")
    manifest = RunManifest(
        run_id=run_id_for(req), created_at=now_iso(), request=req, mode=mode,
        source_runs=sorted(sources.runs.values(), key=lambda r: r.source_id), agent_runs=reports, fields=records,
        cross_checks=checks, labels=run_labels,
        attribution=[SOURCES[s].attribution for s in sorted(used) if s in SOURCES and SOURCES[s].attribution],
        untrusted_text=untrusted, watchdog=watchdog,
    )
    run_dir = write_run(out_root / manifest.run_id, req, manifest, known, by_field)
    progress(f"wrote {run_dir}")
    return run_dir
