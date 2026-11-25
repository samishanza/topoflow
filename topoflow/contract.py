"""Shared contract: how a value enters the record, and what a run writes down about itself.

One owner changes this file (see docs/project/decision.md, D5). Everything else builds against it.
Arrays travel beside these models, never inside them, so a manifest stays small enough to read.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .catalog import DataKind, Shape, Tier

DataOrigin = Literal["real", "synthetic"]
SourceStatus = Literal["ok", "missing", "barred", "error", "skipped"]
FieldState = Literal["observed", "derived", "conflicting", "missing"]
CheckStatus = Literal["pass", "fail", "unknown"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Evidence(BaseModel):
    """Where a value came from. Someone must be able to open `refs` and see the same numbers."""

    source_id: str
    kind: DataKind
    tier: Tier
    refs: list[str]
    label: str
    retrieved_at: str
    data_origin: DataOrigin = "real"
    license: str = ""
    # True when the response was replayed from a recording rather than fetched now.
    replayed: bool = False
    # Free text that arrived with the data (file attributes, names). Data, never instructions.
    untrusted_text: str = ""


class Proposal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_id: str
    field: str
    shape: Shape
    unit: str
    evidence: Evidence | None
    confidence: float
    rationale: str
    # Derived rather than read. Must name the fields it was computed from.
    derived: bool = False
    derived_from: list[str] = Field(default_factory=list)
    # The payload: an ndarray (time, lat, lon) for a grid, a list of features for points, a dict for a scalar.
    value: Any = Field(default=None, exclude=True)


class Rejection(BaseModel):
    agent_id: str
    field: str
    source_id: str | None
    codes: list[str]
    reasons: list[str]


class BudgetBreach(BaseModel):
    agent_id: str
    limit: str
    allowed: float
    used: float


class SourceRun(BaseModel):
    source_id: str
    status: SourceStatus
    reason: str = ""
    calls: int = 0
    bytes: int = 0
    ms: int = 0
    # When barred or missing: what a person can do to open it.
    person_route: str = ""


class AgentRunReport(BaseModel):
    agent_id: str
    tools_called: list[str]
    out_of_scope_tools: list[str]
    accepted: list[str]
    rejected: list[Rejection]
    could_not_resolve: list[str]
    notes: list[str]
    wall_ms: int
    budget_breach: BudgetBreach | None = None


class Candidate(BaseModel):
    source_id: str
    agent_id: str
    confidence: float
    data_origin: DataOrigin
    summary: dict[str, float | int | str | None] = Field(default_factory=dict)


class FieldRecord(BaseModel):
    field: str
    state: FieldState
    unit: str
    selected: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    # When two admitted candidates disagree: the numbers, so a person can judge.
    conflict: dict[str, float | str] | None = None
    note: str = ""


class CrossCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str
    numbers: dict[str, float | int | None] = Field(default_factory=dict)


class RunRequest(BaseModel):
    region: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    resolution_deg: float
    start: str  # YYYY-MM-DD
    end: str
    hours_utc: list[int]


class RunManifest(BaseModel):
    run_id: str
    created_at: str
    request: RunRequest
    mode: Literal["live", "record", "replay"]
    source_runs: list[SourceRun]
    agent_runs: list[AgentRunReport]
    fields: dict[str, FieldRecord]
    cross_checks: list[CrossCheck]
    labels: list[str] = Field(default_factory=list)
    attribution: list[str] = Field(default_factory=list)
    # Text found in source files, by source. Shown to people and the review agent as data. Never obeyed.
    untrusted_text: dict[str, str] = Field(default_factory=dict)
    # Instruction-like passages the watchdog found in that text.
    watchdog: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- payloads sources hand to agents


class SourceDoc(BaseModel):
    """What one source returned, before any agent interprets it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_id: str
    refs: list[str]
    label: str
    retrieved_at: str
    data_origin: DataOrigin = "real"
    replayed: bool = False
    # Free text that came from the source (file attributes, names). Untrusted: never an instruction.
    raw_text: str = ""


class GridPayload(SourceDoc):
    times: Any  # np.ndarray of datetime64[ns]
    lats: Any
    lons: Any
    variables: dict[str, Any]  # name -> ndarray (time, lat, lon)
    units: dict[str, str]


class PointsPayload(SourceDoc):
    features: list[dict[str, Any]]  # {"lon", "lat", "properties": {...}}


class ScalarPayload(SourceDoc):
    values: dict[str, Any]


def grid_summary(arr: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(arr)
    if not finite.any():
        return {"cells": int(arr.size), "finite": 0, "min": None, "max": None, "mean": None}
    a = arr[finite]
    return {
        "cells": int(arr.size),
        "finite": int(finite.sum()),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
        "mean": round(float(a.mean()), 6),
    }
