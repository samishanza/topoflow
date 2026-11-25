"""TopoFlow dashboard: one compiled run, every layer traced to its source.

    python -m app.dashboard                 # newest run under data/runs
    python -m app.dashboard --run <dir>     # a specific run

Reads only what a run wrote (fields.nc, points.geojson, manifest.json). It never fetches, so what you see is what the
gate admitted, and the panels below the map say where each layer came from and what was refused.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

from topoflow.store import latest_run, load_run

FONT = "Arial, sans-serif"
INK, INK2, MUTED, SURFACE, PAGE, RULE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#f7f9fc", "#e5e7eb"
# Categorical slots 1-3, validated all-pairs for a scatter (light: CVD dE 9.2, normal 24.0). Aqua sits under 3:1
# against the surface, so every point also carries its family in shape, hover and the table below the map.
FAMILY = {"Node": "#2a78d6", "Saddle": "#eb6834", "Focus": "#1baf7a"}
SYMBOL = {"saddle": "x", "attracting_node": "circle", "repelling_node": "circle-open",
          "attracting_focus": "diamond", "repelling_focus": "diamond-open", "center": "star"}
# Emitters and plumes are told apart by shape, in neutral ink: six hues on one scatter fail the CVD check.
EMITTER = {"emitters.assets": ("square-open", "Climate TRACE asset"),
           "emitters.facilities": ("triangle-up-open", "EPA GHGRP facility"),
           "plumes.detections": ("star-triangle-up", "Carbon Mapper plume")}
# Single-hue sequential for magnitude; blue<->red with a gray midpoint for signed fields.
SEQUENTIAL = "Purples"
DIVERGING = [[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#c23b3b"]]

LAYERS = {
    "ch4__surface_ppb": ("Surface CH4", "ppb", SEQUENTIAL, False),
    "ch4__column_ppb": ("TROPOMI XCH4 (column)", "ppb", SEQUENTIAL, False),
    "ch4__emission_flux": ("EDGAR CH4 emission flux (log10)", "log10 kg m-2 s-1", SEQUENTIAL, False),
    "met__blh": ("Boundary layer height", "m", SEQUENTIAL, False),
    "flow__divergence": ("Divergence of 10 m wind", "10^-5 s-1", DIVERGING, True),
    "flow__vorticity": ("Relative vorticity", "10^-5 s-1", DIVERGING, True),
}

STATUS_STYLE = {"ok": "#0ca30c", "pass": "#0ca30c", "missing": "#b26b00", "skipped": MUTED, "unknown": MUTED,
                "barred": "#b26b00", "error": "#c23b3b", "fail": "#c23b3b"}


def _status_badge(s: str):
    icon = {"ok": "✓", "pass": "✓", "fail": "✕", "error": "✕", "missing": "!", "barred": "🔒", "skipped": "–",
            "unknown": "?"}.get(s, "")
    return html.Span(f"{icon} {s}", style={"color": STATUS_STYLE.get(s, INK2), "fontWeight": "600",
                                           "whiteSpace": "nowrap"})


def build_app(run_dir: Path) -> dash.Dash:
    manifest, ds, pts = load_run(run_dir)
    times = pd.DatetimeIndex(ds["time"].values)
    lats, lons = ds["lat"].values, ds["lon"].values
    res = float(manifest.request.resolution_deg)
    layers = {k: v for k, v in LAYERS.items() if k in ds}
    feats = pts["features"]
    cps = [f for f in feats if f["properties"]["field"] == "topology.critical_points"]
    coast = [f for f in feats if f["properties"]["field"] == "static.coastline"]
    u = ds["wind__u10"].values if "wind__u10" in ds else None
    v = ds["wind__v10"].values if "wind__v10" in ds else None
    speed95 = float(np.nanpercentile(np.hypot(u, v), 95)) if u is not None else 1.0

    # Coastline as one trace: segments separated by None.
    cx, cy = [], []
    for f in coast:
        cx += f["properties"]["x"] + [None]
        cy += f["properties"]["y"] + [None]

    app = dash.Dash(__name__, title="TopoFlow")
    banner = [html.Div(label, style={"background": "#fff4e5", "border": "1px solid #f0c27a", "color": "#5c3b00",
                                     "padding": "8px 12px", "borderRadius": "8px", "marginBottom": "8px",
                                     "fontSize": "13px"})
              for label in manifest.labels]
    banner += [html.Div(f"Watchdog: {w}", style={"background": "#fdecec", "border": "1px solid #e9a3a3",
                                                  "color": "#6b1111", "padding": "8px 12px", "borderRadius": "8px",
                                                  "marginBottom": "8px", "fontSize": "13px"})
               for w in manifest.watchdog]

    card = {"padding": "16px 20px", "backgroundColor": "white", "borderRadius": "12px", "marginTop": "14px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.04)"}
    label_style = {"fontWeight": "600", "fontSize": "14px", "color": "#374151", "marginRight": "10px"}
    h2 = {"fontSize": "17px", "fontWeight": "600", "color": "#1f2937", "margin": "0 0 10px"}
    req = manifest.request

    source_rows = [{"source": s.source_id, "status": s.status, "calls": s.calls, "MB": round(s.bytes / 1e6, 2),
                    "reason / how to open": s.reason + (f" -> {s.person_route}" if s.person_route and s.status != "ok" else "")}
                   for s in manifest.source_runs]
    field_rows = [{"field": k, "state": f.state, "chosen source": f.selected or "",
                   "candidates": ", ".join(c.source_id for c in f.candidates),
                   "agreement": (f"{f.conflict.get('verdict', 'CONFLICT')} vs {f.conflict['against']} "
                                 f"(rel. RMS {f.conflict['relative_rms']})") if f.conflict else ""}
                  for k, f in manifest.fields.items()]
    refused_rows = [{"agent": a.agent_id, "field": r.field, "source": r.source_id or "", "codes": ", ".join(r.codes),
                     "why": "; ".join(r.reasons)} for a in manifest.agent_runs for r in a.rejected]
    cp_rows = [{"time": f["properties"]["time"][:16], "kind": f["properties"]["kind"].replace("_", " "),
                "lat": f["geometry"]["coordinates"][1], "lon": f["geometry"]["coordinates"][0],
                "strength (1e-5/s)": round(f["properties"]["strength"] * 1e5, 2),
                "CH4 anomaly (ppb)": f["properties"].get("ch4_anomaly_ppb"), "BLH (m)": f["properties"].get("blh_m")}
               for f in cps]

    table_style = dict(style_table={"overflowX": "auto"}, page_size=12,
                       style_cell={"fontFamily": FONT, "fontSize": "12px", "textAlign": "left", "padding": "6px",
                                   "whiteSpace": "normal", "height": "auto", "maxWidth": "520px", "color": INK},
                       style_header={"fontWeight": "600", "backgroundColor": PAGE, "color": "#374151"})

    app.layout = html.Div(style={"fontFamily": FONT, "backgroundColor": PAGE, "padding": "22px", "minHeight": "100vh"}, children=[
        html.H1("TopoFlow: Wind Topology + CH4", style={"fontWeight": "600", "fontSize": "28px", "color": "#1f2937",
                                                        "textAlign": "center", "margin": "0 0 4px"}),
        html.P(f"{req.region} · {req.bbox} · {req.resolution_deg}° grid · {req.start} to {req.end} · run {manifest.run_id}",
               style={"fontSize": "13px", "color": "#6b7280", "textAlign": "center", "margin": "0 0 14px"}),
        *banner,
        html.Div(style={**card, "display": "flex", "flexWrap": "wrap", "gap": "18px", "alignItems": "center", "marginTop": 0}, children=[
            html.Div([html.Label("Layer", style=label_style),
                      dcc.Dropdown(id="layer", options=[{"label": v[0], "value": k} for k, v in layers.items()],
                                   value=next(iter(layers), None), clearable=False, style={"width": "290px"})],
                     style={"display": "flex", "alignItems": "center"}),
            dcc.Checklist(id="overlays", value=["wind", "points", "emitters"], inline=True,
                          options=[{"label": " Wind", "value": "wind"}, {"label": " Critical points", "value": "points"},
                                   {"label": " Emitters & plumes", "value": "emitters"}],
                          inputStyle={"marginLeft": "12px"}, style={"fontSize": "14px", "color": "#374151"}),
            dcc.Checklist(id="kinds", inline=True, value=list(SYMBOL),
                          options=[{"label": " " + k.replace("_", " "), "value": k} for k in SYMBOL],
                          inputStyle={"marginLeft": "10px"}, style={"fontSize": "12px", "color": INK2}),
        ]),
        dcc.Graph(id="map", style={"height": "70vh", "backgroundColor": "white", "borderRadius": "12px", "marginTop": "14px"},
                  config={"displayModeBar": True, "scrollZoom": True}),
        html.Div(style=card, children=[
            html.Label("Time (UTC)", style=label_style),
            dcc.Slider(id="t", min=0, max=len(times) - 1, value=0, step=1,
                       marks={i: {"label": times[i].strftime("%m-%d %Hh"), "style": {"fontSize": "11px", "color": "#6b7280"}}
                              for i in range(0, len(times), max(1, len(times) // 10))},
                       tooltip={"placement": "bottom"}),
        ]),
        html.Div(style=card, children=[html.H2("Cross-checks", style=h2), html.Div([
            html.Div([_status_badge(c.status), html.Span(f"  {c.name}", style={"fontWeight": "600", "color": INK}),
                      html.Div(c.detail, style={"color": INK2, "fontSize": "13px", "margin": "2px 0 10px 18px"})])
            for c in manifest.cross_checks])]),
        html.Div(style=card, children=[html.H2("Fields: where each layer came from", style=h2),
                                       dash_table.DataTable(field_rows, **table_style)]),
        html.Div(style=card, children=[html.H2("Sources", style=h2), dash_table.DataTable(source_rows, **table_style)]),
        html.Div(style=card, children=[html.H2(f"Refused by the gate ({len(refused_rows)})", style=h2),
                                       dash_table.DataTable(refused_rows, **table_style) if refused_rows
                                       else html.P("Nothing was refused.", style={"color": INK2})]),
        html.Div(style=card, children=[html.H2(f"Critical points ({len(cp_rows)}), all times", style=h2),
                                       html.P("Inferred from the sampled grid: a point can move or vanish at a finer resolution.",
                                              style={"color": INK2, "fontSize": "13px"}),
                                       dash_table.DataTable(cp_rows, sort_action="native", filter_action="native", **table_style)]),
        html.Div(style={**card, "fontSize": "12px", "color": INK2}, children=[
            html.H2("Attribution", style=h2), *[html.P(a, style={"margin": "2px 0"}) for a in manifest.attribution]]),
    ])

    @app.callback(Output("map", "figure"), Input("t", "value"), Input("layer", "value"),
                  Input("overlays", "value"), Input("kinds", "value"))
    def draw(ti, layer, overlays, kinds):
        fig = go.Figure()
        if layer:
            title, unit, scale, signed = layers[layer]
            da = ds[layer]
            z = da.values if "time" not in da.dims else da.isel(time=ti).values
            if layer == "ch4__emission_flux":
                z = np.log10(np.where(z > 0, z, np.nan))
            if signed:
                z = z * 1e5
                lim = float(np.nanpercentile(np.abs(z), 98)) or 1.0
            src = da.attrs.get("source", "")
            fig.add_trace(go.Heatmap(
                z=z, x=lons, y=lats, colorscale=scale, opacity=0.85, zmid=0 if signed else None,
                zmin=-lim if signed else None, zmax=lim if signed else None,
                colorbar=dict(title=dict(text=unit, font=dict(family=FONT, size=12, color=INK2)),
                              tickfont=dict(family=FONT, size=11, color=INK2), thickness=12, len=0.8),
                hovertemplate=f"{title}<br>%{{y:.2f}}°, %{{x:.2f}}°<br>%{{z:.4g}} {unit}<br>source: {src}<extra></extra>",
                name=title))
        if cx:
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=INK2, width=1), hoverinfo="skip",
                                     showlegend=False))
        if "wind" in overlays and u is not None:
            uu, vv = u[ti], v[ti]
            step = max(1, int(round(max(len(lats), len(lons)) / 45)))
            la, lo = np.meshgrid(lats[::step], lons[::step], indexing="ij")
            us, vs = uu[::step, ::step], vv[::step, ::step]
            k = 0.9 * res * step / max(speed95, 0.1)
            x1 = lo + us * k / np.maximum(np.cos(np.deg2rad(la)), 0.2)
            y1 = la + vs * k
            ok = np.isfinite(us) & np.isfinite(vs)
            xs = np.column_stack([lo[ok], x1[ok], np.full(ok.sum(), np.nan)]).ravel()
            ys = np.column_stack([la[ok], y1[ok], np.full(ok.sum(), np.nan)]).ravel()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(11,11,11,0.55)", width=1.2),
                                     hoverinfo="skip", name="10 m wind", showlegend=True))
            sp = np.hypot(us[ok], vs[ok])
            ang = (np.rad2deg(np.arctan2(us[ok], vs[ok])) + 360) % 360  # toward, clockwise from north
            fig.add_trace(go.Scatter(x=x1[ok], y=y1[ok], mode="markers", showlegend=False,
                                     marker=dict(symbol="arrow", size=7, angle=ang, color="rgba(11,11,11,0.7)"),
                                     customdata=np.column_stack([sp, (ang + 180) % 360]),
                                     hovertemplate="wind %{customdata[0]:.1f} m/s from %{customdata[1]:.0f}°<extra></extra>"))
        if "points" in overlays:
            now = times[ti]
            here = [f for f in cps if pd.Timestamp(f["properties"]["time"]) == now and f["properties"]["kind"] in kinds]
            for kind in SYMBOL:
                fs = [f for f in here if f["properties"]["kind"] == kind]
                if not fs:
                    continue
                fam = fs[0]["properties"]["family"]
                cd = [[f["properties"]["strength"] * 1e5, f["properties"].get("ch4_anomaly_ppb"),
                       f["properties"].get("blh_m")] for f in fs]
                fig.add_trace(go.Scatter(
                    x=[f["geometry"]["coordinates"][0] for f in fs], y=[f["geometry"]["coordinates"][1] for f in fs],
                    mode="markers", name=kind.replace("_", " "), legendgroup=fam,
                    marker=dict(symbol=SYMBOL[kind], size=11, color=FAMILY[fam], line=dict(color=SURFACE, width=2)
                                if not SYMBOL[kind].endswith("open") else dict(color=FAMILY[fam], width=2.5)),
                    customdata=cd,
                    hovertemplate=(f"<b>{kind.replace('_', ' ')}</b> ({fam})<br>%{{y:.2f}}°, %{{x:.2f}}°"
                                   "<br>strength %{customdata[0]:.2f}e-5 /s<br>CH4 anomaly %{customdata[1]} ppb"
                                   "<br>BLH %{customdata[2]} m<br><i>inferred</i><extra></extra>")))
        if "emitters" in overlays:
            day = times[ti].normalize()
            for key, (sym, name) in EMITTER.items():
                fs = [f for f in feats if f["properties"]["field"] == key]
                if key == "plumes.detections":
                    fs = [f for f in fs if f["properties"].get("time") and
                          pd.Timestamp(f["properties"]["time"]).tz_localize(None).normalize() == day]
                if not fs:
                    continue
                val = {"emitters.assets": "t_ch4_per_year", "emitters.facilities": "t_co2e_ch4_per_year",
                       "plumes.detections": "kg_per_hour"}[key]
                unit = {"emitters.assets": "t CH4/yr", "emitters.facilities": "t CO2e/yr", "plumes.detections": "kg/h"}[key]
                vals = np.array([f["properties"].get(val) or 0 for f in fs], float)
                size = 5 + 7 * np.clip(np.log10(np.maximum(vals, 1)) / max(np.log10(vals.max() + 1), 1), 0, 1)
                fig.add_trace(go.Scatter(
                    x=[f["geometry"]["coordinates"][0] for f in fs], y=[f["geometry"]["coordinates"][1] for f in fs],
                    mode="markers", name=name, marker=dict(symbol=sym, size=size, color=INK2, opacity=0.45,
                                                           line=dict(width=1, color=INK2)),
                    text=[str(f["properties"].get("name") or f["properties"].get("plume_id") or "") for f in fs],
                    customdata=vals,
                    hovertemplate=f"<b>%{{text}}</b><br>{name}<br>%{{customdata:,.0f}} {unit}<extra></extra>"))
        w, s, e, n = manifest.request.bbox
        pad = res
        fig.update_layout(
            font=dict(family=FONT, size=12, color="#374151"),
            title=dict(text=f"{layers[layer][0] if layer else 'Wind'} · {times[ti]:%Y-%m-%d %H:%M} UTC",
                       font=dict(family=FONT, size=17, color="#1f2937"), x=0.5, xanchor="center"),
            xaxis=dict(title="Longitude", range=[w - pad, e + pad], gridcolor=RULE, zeroline=False,
                       tickfont=dict(size=11, color="#6b7280")),
            yaxis=dict(title="Latitude", range=[s - pad, n + pad], scaleanchor="x",
                       scaleratio=1 / max(np.cos(np.deg2rad((s + n) / 2)), 0.3), gridcolor=RULE, zeroline=False,
                       tickfont=dict(size=11, color="#6b7280")),
            plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=60, r=40, t=55, b=50),
            legend=dict(orientation="h", y=-0.12, font=dict(size=12), bgcolor="rgba(255,255,255,0.8)"),
            hoverlabel=dict(font=dict(family=FONT, size=12)),
            uirevision=layer,  # keep zoom when only the time changes
        )
        return fig

    app.draw = draw  # the figure without a server, for tests and previews
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=os.environ.get("TOPOFLOW_RUN"))
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    run = Path(a.run) if a.run else latest_run(a.runs)
    if not run:
        raise SystemExit("No runs yet. Run `python -m topoflow collect` first, or replay the recorded one: "
                         "`python -m topoflow collect --mode replay`.")
    print(f"serving {run}")
    build_app(run).run(debug=False, port=a.port)


if __name__ == "__main__":
    main()
