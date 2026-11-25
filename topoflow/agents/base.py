"""Agents search and read. They never decide, and they never write.

Each agent returns proposals. The gatekeeper admits or rejects each one; the reconciler picks among what is admitted;
deterministic code derives everything else. An agent that cannot find a value says so with miss().
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from .. import grid
from ..catalog import is_agent_allowed, spec_for
from ..contract import Evidence, GridPayload, Proposal, RunRequest, SourceDoc
from ..sources import SourceClient
from ..sources.registry import SOURCES


_INSTRUCTION = re.compile(
    r"\b(ignore (all|any|the)? ?(previous|prior|above)|system (note|prompt|message)|you (must|should) now|"
    r"disregard|mark (it|this|the \w+|every \w+) as (verified|passed|approved)|set \S+( \S+)? to \d)",
    re.IGNORECASE,
)


def looks_like_instruction(text: str) -> bool:
    """A cheap, deterministic watchdog. It flags; it never filters or acts. False positives just add a note."""
    return bool(_INSTRUCTION.search(text or ""))


@dataclass
class AgentContext:
    request: RunRequest
    sources: SourceClient
    # Values admitted so far, by field. Lets an agent skip work another source already did.
    known: dict[str, Any] = field(default_factory=dict)
    # The running agent's declared tools. A call outside them is refused and reported, never made.
    allowed_tools: tuple[str, ...] = ()


class Agent(Protocol):
    id: str
    purpose: str
    # The only tools this agent may call. Calling anything else is a watchdog alert.
    tools: tuple[str, ...]

    def run(self, ctx: AgentContext) -> "AgentResult": ...


@dataclass
class AgentResult:
    agent_id: str
    proposals: list[Proposal]
    tool_calls: list[str]
    could_not_resolve: list[str]
    notes: list[str]


class Emit:
    """Collects an agent's output. propose() needs the source document, so no value exists without evidence."""

    def __init__(self, agent_id: str, ctx: AgentContext):
        self.agent_id = agent_id
        self.ctx = ctx
        self.proposals: list[Proposal] = []
        self.tool_calls: list[str] = []
        self.missed: list[str] = []
        self.notes: list[str] = []

    def call(self, tool: str, **kwargs):
        self.tool_calls.append(tool)
        if self.ctx.allowed_tools and tool not in self.ctx.allowed_tools:
            self.notes.append(f"refused {tool}: not on {self.agent_id}'s tool list")
            return None
        return self.ctx.sources.call(tool, **kwargs)

    def evidence(self, doc: SourceDoc) -> Evidence:
        entry = SOURCES.get(doc.source_id)
        kind = "synthetic" if doc.data_origin == "synthetic" else (entry.kind if entry else "local_file")
        if doc.raw_text and looks_like_instruction(doc.raw_text):
            note = f"instruction-like text in {doc.source_id}; kept as data and not followed"
            if note not in self.notes:
                self.notes.append(note)
        return Evidence(source_id=doc.source_id, kind=kind, tier=entry.tier if entry else 3, refs=doc.refs,
                        label=doc.label, retrieved_at=doc.retrieved_at, data_origin=doc.data_origin,
                        license=entry.license if entry else "", replayed=doc.replayed,
                        untrusted_text=doc.raw_text[:2000])

    def propose(self, key: str, value: Any, doc: SourceDoc | None, unit: str, *, confidence: float, rationale: str,
                derived: bool = False, derived_from: list[str] | None = None) -> bool:
        if not is_agent_allowed(self.agent_id, key):
            self.notes.append(f"skipped {key}: outside {self.agent_id}'s scope")
            return False
        if value is None:
            return False
        spec = spec_for(key)
        self.proposals.append(Proposal(
            agent_id=self.agent_id, field=key, shape=spec.shape if spec else "grid", unit=unit,
            evidence=self.evidence(doc) if doc is not None else None, confidence=confidence, rationale=rationale,
            derived=derived, derived_from=derived_from or [], value=value,
        ))
        return True

    def miss(self, *keys: str):
        for k in keys:
            if k not in self.missed:
                self.missed.append(k)

    def note(self, text: str):
        self.notes.append(text)

    def result(self) -> AgentResult:
        return AgentResult(self.agent_id, self.proposals, self.tool_calls, self.missed, self.notes)


def align_to_run(payload: GridPayload, arr: np.ndarray, req: RunRequest) -> np.ndarray:
    """Put a (t, lat, lon) array on the run's times. A daily field covers every run time on its day."""
    want = pd.DatetimeIndex(grid.times(req))
    have = pd.DatetimeIndex(payload.times)
    if len(have) == len(want) and (have == want).all():
        return arr
    days = have.normalize()
    idx = days.get_indexer(want.normalize())
    out = np.full((len(want),) + arr.shape[1:], np.nan, dtype="float32")
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out


AgentFactory = Callable[[], Agent]
