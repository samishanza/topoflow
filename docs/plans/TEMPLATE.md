# Plan: <name>

Copy to `docs/plans/<yyyy-mm-dd>-<name>.md`. Delete any section that does not apply.

- **Status:** Draft | Agreed | In progress | Integrated
- **Lead:** <one person coordinating>
- **Shared-contract owner:** <one person; the only one who changes catalog.py, contract.py, and dependencies>
- **Baseline commit:** <sha all worktrees start from>
- **Decision link:** `docs/project/decision.md` <decision IDs>

## Demo path

| Step | What happens | Evidence the step worked |
|---|---|---|
| Collect | | |
| Show a field, a refusal, or a cross-check | | |
| Person inspects and decides | | |
| State change | | |

## Non-goals

- <what this plan will not do>

## Data model changes

<fields added to catalog.py, with unit, range, allowed agents, preference; contract changes>

## Fixture

- **Path:** <fixtures/... or the synthetic client>
- **Labelled synthetic in:** <data_origin, manifest labels, dashboard banner>
- **Expected result:** <exact, checkable output>

## Shared interfaces

Agree on and commit these before parallel work starts.

| Interface | Path | Sample | Owner |
|---|---|---|---|
| | | | |

## Checks that prove it works

| Check | Exact command | Expected result |
|---|---|---|
| | | |

## Tasks

| ID | Objective (one) | Owner (one) | Depends on | Allowed files | Acceptance check and expected result | Handoff destination | Blockers | Status |
|---|---|---|---|---|---|---|---|---|
| T1 | | | none | | | | | Not started |

Integration order: <T1, then T2, ...>

## Worker lanes

| Lane | Scope | Runs as (ce-work worker, agent teammate, worktree session, or subagent) | Worktree |
|---|---|---|---|
| Sources | Adapters, registry entries, recorded fixtures, terms notes | | |
| Gate and derive | Catalog bounds, gate checks, derivations, cross-checks, tests | | |
| Dashboard | Layers, overlays, provenance panels, rendered checks | | |
| Lead/integration | Shared contract, merge order, demo QA | | |

## Worker brief

```text
Plan path:
Task:
Branch or worktree:
Start from baseline:
Allowed files:
Shared contract (paths): topoflow/catalog.py, topoflow/contract.py
Code examples (paths):
Fixtures (paths):
Acceptance check and expected result:
Handoff destination:
Do not: change the shared contract; add dependencies; commit Carbon Mapper responses; run a second live collection
while another is running (shared Open-Meteo budget)
```

## Handoffs

```text
Task:
Branch or commit:
Changed files:
Commands run:
Results:
Blockers:
Next action:
Decisions to record in decision.md:
```

## Integration report

- **Combined diff reviewed by:**
- **Replay of the demo run:**
- **Person can inspect evidence, see refusals and cross-checks, and record a decision:**
- **Checks run:**
- **Remaining issues:**
