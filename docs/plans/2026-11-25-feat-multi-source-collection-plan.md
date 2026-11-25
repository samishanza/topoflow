# Plan: multi-source collection behind a deterministic gate

- **Status:** In progress (T1-T9 done; T10-T14 open)
- **Lead:** session on `main`
- **Shared-contract owner:** session on `main`
- **Baseline commit:** the commit that adds this file
- **Decision link:** [decision.md](../project/decision.md) D1-D11; open D12-D14

## Demo path

| Step | What happens | Evidence the step worked |
|---|---|---|
| Collect | `python -m topoflow collect --mode replay` rebuilds the Permian run from recorded public data, no network | `show` lists 7+ sources `ok`, fields observed/derived, cross-checks with numbers |
| Show an issue | The dashboard draws CH4, wind, critical points and emitters; the panels show each layer's source, the refusals and the cross-checks | A plume-wind check that passes "too well" carries its caveat; inland coastline is a note, not a failure |
| Oversight | The synthetic doctored file (in-range cells, impossible regional mean, embedded instruction) is refused on cross-field physics and flagged by the watchdog | `tests/test_pipeline.py::test_doctored_file_is_refused_flagged_and_changes_nothing` |
| Person decides | `topoflow review` (Claude) proposes flags; `topoflow decide <id> accept|reject --by --reason` records an attributed, final decision | `review_decisions.jsonl` beside the run |

## Non-goals

- Facility-level emission estimates, inversions, or compliance judgements.
- A hypothesis test of CH4 at critical points (D12 must be designed first).
- A database or hosted deployment.

## Checks that prove it works

| Check | Exact command | Expected result |
|---|---|---|
| Unit, gate, topology, pipeline, review tests | `python3 -m pytest` | all pass, no network |
| Replay | `python3 -m topoflow collect --mode replay` | Permian run, ERA5 vs POWER `pass` |
| Topology vs resolution | `python3 scripts/eval_topology.py` | recall ~0.93 at 0.5 deg, ~0.47 at 2.5 deg |
| Dashboard | `python3 -m app.dashboard` | serves the newest run on :8050 |

## Tasks

| ID | Objective | Allowed files | Acceptance check | Status |
|---|---|---|---|---|
| T1 | Shared contract: catalog with units, ranges, scopes, preferences; Evidence/Proposal/Manifest | `topoflow/catalog.py`, `topoflow/contract.py` | imported by every module | Done |
| T2 | Fetcher with live / record (resumable) / replay, per-provider rate budgets | `topoflow/http.py` | replay test passes offline | Done |
| T3 | Adapters: Open-Meteo ERA5 + CAMS, NASA POWER, NOAA GML, Climate TRACE, EPA GHGRP, Carbon Mapper, EDGAR, Natural Earth | `topoflow/sources/*` | Permian record run: each `ok` | Done (tested live) |
| T4 | Keyed/local adapters: CDS ERA5, Earth Engine TROPOMI, local ERA5 and CH4 | `topoflow/sources/keyed.py` | barred with a person route when no key | CDS tested live; GEE and local untested |
| T5 | Agents with scoped tools; out-of-scope calls refused | `topoflow/agents/*` | `test_agent_cannot_call_tools_off_its_list` | Done |
| T6 | Gate, budgets, disk guard | `topoflow/gatekeeper/*` | `tests/test_gatekeeper.py` | Done |
| T7 | Reconcile, derive, topology, cross-checks, store | `topoflow/derive.py`, `topology.py`, `crosschecks.py`, `store.py` | synthetic run finds the planted points | Done |
| T8 | Claude review agent and attributed decisions | `topoflow/review.py`, `topoflow/cli.py` | `tests/test_review.py` (stub client) | Done; not yet run against the live API |
| T9 | Dashboard over a run, rendered and checked | `app/dashboard.py` | screenshots of CH4 and divergence layers | Done |
| T10 | Global run with CAMS after the rate-budget fix | none (run only) | CAMS `ok` in the global manifest | See Handoffs |
| T11 | Earth Engine TROPOMI live test (D13) | `topoflow/sources/keyed.py` | `ch4.column_ppb` observed for the Permian week | Blocked on a registered Cloud project |
| T12 | CAMS EAC4 via ADS for pre-2025 windows (D14) | new adapter + registry entry | a 2023 run with surface CH4 | Not started |
| T13 | Hypothesis design for CH4 at critical points (D12) | `docs/research/` | written test plan before new data | Not started |
| T14 | First live review agent run, decisions recorded | none | `review_proposals.jsonl` with validated proposals | Needs `ANTHROPIC_API_KEY` |

## Handoffs

```text
Task: T10
Results: see docs/project/data-flow.md "First runs" and the run folder named in the latest `topoflow show`.
Next action: if CAMS is still rate-limited, rerun the same command later; record mode resumes from what was saved.
```
