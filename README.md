# TopoFlow

Where does near-surface air converge, and is methane higher there? TopoFlow compiles wind, methane, boundary-layer,
emission-inventory and plume data from a dozen public sources onto one grid and one clock, finds the critical points
of the wind field (nodes, saddles, foci), and draws it all in a Dash dashboard. Every value traces to its source.

It is built the way a careful team builds a data product with agents:

- **Agents read and propose.** Four agents (context, wind, methane, emissions) each own a slice of the field catalog and
  a scoped tool list. A proposal without evidence cannot be made.
- **A deterministic gate admits.** Unit, shape, missing-data share, physical range and cross-field physics. Refusals
  are kept, with codes and reasons.
- **Code decides.** Reconciliation among sources, CH4 unit conversion, divergence, vorticity, critical points and
  cross-checks are tested code. No language model produces a number in the record.
- **Claude reviews, a person decides.** An optional review agent reads a finished run and proposes flags; a named
  person accepts or rejects each, and the decision is final and attributed.

Research prototype: no facility emission estimates, compliance judgements or health claims.

## Quick start (macOS)

```bash
git clone https://github.com/samishanza/topoflow.git && cd topoflow
python3 -m pip install -e ".[dev]"            # numpy, xarray, netCDF4, pandas, shapely, pydantic, dash, plotly, pytest

python3 -m pytest                              # 63 tests, no network
python3 -m topoflow collect --mode replay      # rebuild the recorded Permian run offline
python3 -m app.dashboard                       # http://127.0.0.1:8050
```

Live collection (uses free public APIs; about 1 to 2 minutes for the Permian default):

```bash
python3 -m topoflow collect                                   # Permian, 0.5 deg, 2025-07-01..07
python3 -m topoflow collect --region global                   # 5 deg; about 5,000 Open-Meteo calls
python3 -m topoflow collect --bbox -98,26,-88,32 --resolution 0.25 --start 2025-08-01 --end 2025-08-03 --hours 0,3,6,9,12,15,18,21
python3 -m topoflow show                                      # sources, fields, refusals, cross-checks of the newest run
python3 -m topoflow sources                                   # every source, its licence, and what it needs
```

Regions: `permian`, `gulf_coast`, `turkmenistan`, `sudd`, `global`, or any `--bbox west,south,east,north`.
CAMS methane on Open-Meteo starts in early 2025, so earlier windows have no surface CH4 unless you add your own files.

### Optional sources and the review agent

Copy `.env.example` to `.env` and fill in what you have. Each keyed source reports itself `barred`, with the route to
open it, until its credentials exist.

| Source | Needs |
|---|---|
| Copernicus CDS ERA5 (native u/v) | `~/.cdsapirc` and `pip install cdsapi` |
| TROPOMI XCH4 via Earth Engine | `EARTHENGINE_PROJECT`, `earthengine authenticate`, `pip install earthengine-api` |
| Your own ERA5 / CH4 NetCDF | `TOPOFLOW_U_FILE`, `TOPOFLOW_V_FILE`, `TOPOFLOW_METHANE_DIR` (one `YYYYMMDD.nc` per day) |
| Claude review agent | `ANTHROPIC_API_KEY` and `pip install anthropic` |

```bash
python3 -m topoflow review                                    # Claude reads the newest run and proposes flags
python3 -m topoflow decide                                    # list proposals
python3 -m topoflow decide <id> reject --by "Your Name" --reason "BLH explains it"
```

The review agent uses `claude-opus-5-5` at effort `high`, with server-side refusal fallback enabled. Each review costs
API tokens.

## The globe: stagnation points over whole planets

```bash
python3 scripts/fetch_era5_arco.py --start 2019-01 --end 2019-12   # stream ERA5 from ARCO-ERA5 (public, no key)
python3 -m topoflow.climatology process --watch                    # gate, analyse, track, store each month
python3 scripts/fetch_ncep.py --product r1 --years 2019             # optional: NCEP R1 / R2 for cross-product checks
python3 -m topoflow.mars --year 34                                  # optional: Mars from OpenMARS
python3 -m app.globe_server                                         # http://127.0.0.1:8060
```

Every 6-hourly step is analysed over the whole sphere (poles included): stagnation points are located on the
piecewise-linear wind, classified by the Jacobian (saddle, attracting/repelling node and focus, centre), and the
Poincaré-Hopf closure is checked. The identity `sum of indices = 2 + edge_error` holds on every step; `edge_error`
counts the grid edges where the wind turns too sharply between neighbouring points for the grid to resolve.
Wind particles on the globe are advected by the real field of the shown step at true speed times a stated time-lapse.
`python3 -m topoflow.tdl.train` trains the simplicial network (TopoNet) against persistence, climatology, logistic and
vertex-only baselines. First results: [docs/research/2026-11-25-closure-mars-tdl-first-results.md](docs/research/2026-11-25-closure-mars-tdl-first-results.md).

## What a run contains

`data/runs/<run_id>/` holds `fields.nc` (every admitted grid, the chosen one and its alternatives), `points.geojson`
(emitters, plumes, critical points, coastline), and `manifest.json` (request, source outcomes, every agent's admitted
and refused proposals, field states, cross-checks, labels, attribution, untrusted text found in sources).

Fields are `observed`, `derived`, `conflicting` or `missing`, and those states stay apart. See
[docs/project/data-flow.md](docs/project/data-flow.md) for the full flow and the first runs' results.

## Sources

ERA5 (Open-Meteo and Copernicus CDS), CAMS global CH4 (Open-Meteo), NASA POWER (MERRA-2), NOAA GML global CH4, Climate
TRACE, EPA GHGRP, Carbon Mapper, EDGAR v8.0, Natural Earth, Sentinel-5P TROPOMI (Earth Engine), and your own files.
Licences, test status and coverage: [docs/research/2026-11-25-ch4-wind-datasets.md](docs/research/2026-11-25-ch4-wind-datasets.md).

Carbon Mapper data is non-commercial and share-alike, and Open-Meteo's free API and Earth Engine are non-commercial.
This repo is private research; re-read those terms before making it public or using it commercially.

## Security notes

- Both servers (`app.dashboard`, `app.globe_server`) bind to `127.0.0.1` and run without debug mode. They read local
  files only and fetch nothing from the browser's requests. Don't expose them on a public interface without a
  production WSGI server in front.
- Keys live in `.env` (gitignored) or the providers' own config files (`~/.cdsapirc`); none are read from the repo.
- Text inside downloaded files is treated as data. The review agent is told never to follow it, and a watchdog flags
  instruction-like text.
- Dependency floors in `pyproject.toml` include fixes for every advisory `pip-audit` reported on 2026-11-25. Run
  `pip-audit` before releasing. Report vulnerabilities through GitHub's private security advisories on this repo.

## Licence

Source available for viewing only. Copyright (c) 2026 Shanza Sami, all rights reserved: no permission is granted to
use, copy, modify or distribute this code without written permission. See [LICENSE](LICENSE).

## Third-party data

See [NOTICE.md](NOTICE.md) for the licence and attribution of every bundled file and every source fetched at run time.

## Join the workflow (Claude Code)

1. Open the repo in Claude Code and accept the prompt to trust it. `.claude/settings.json` enables agent teams and the
   Compound Engineering plugin. If the plugin does not load:

   ```text
   /plugin marketplace add EveryInc/compound-engineering-plugin
   /plugin install compound-engineering
   ```

2. Read, in order: [CLAUDE.md](CLAUDE.md) (the rules), [docs/project/context.md](docs/project/context.md),
   [docs/project/decision.md](docs/project/decision.md), and the current plan in [docs/plans/](docs/plans/).
3. Take a task from the plan.

**The loop:** `/ce-brainstorm`, `/ce-plan`, `/ce-work`, `/ce-simplify-code`, `/ce-code-review`, `/ce-compound`.

| Use | When |
|---|---|
| `/ce-work` isolated workers | The task can run without your steering |
| Named worktree session | You want to steer it yourself |
| Agent team | Workers need a shared task list or to talk to each other |
| Subagents (`source-scout`, `data-qa`) | Bounded research, or checking a run |

Worktree for a task, from the plan's baseline commit:

```bash
git worktree add -b <name>/<task> .worktrees/<task> <baseline-commit>
cd .worktrees/<task> && claude
```

Or one session per terminal: `claude --worktree sources --name sources`, `claude --worktree dashboard --name dashboard`,
`claude --name lead`. Worktrees separate files, not rate budgets: only one session runs a live collection at a time.

**Handing off:** change only your task's files; ask the contract owner to change `catalog.py` or `contract.py`; run your
acceptance check; record branch, files, commands, results, blockers and next action in the plan's Handoffs section.

## Layout

```
topoflow/            catalog, contract, units, grid, http, topology, derive, crosschecks, store, orchestrator, review, cli
topoflow/sources/    one adapter per dataset, registry (licences), synthetic fixture
topoflow/agents/     context, wind, methane, emissions agents
topoflow/gatekeeper/ admissibility checks, budgets, disk guard
app/dashboard.py     the dashboard; app/legacy_dashboard.py is the original single-file version
scripts/             eval_topology.py (detector vs resolution)
fixtures/            recorded public responses for the Permian demo (Carbon Mapper excluded by licence)
docs/                project context and decisions, plans, research
```

The workflow layer is modeled on the agent orchestration in
[maximozhang/MIT-Energy-Hacks](https://github.com/maximozhang/MIT-Energy-Hacks).
