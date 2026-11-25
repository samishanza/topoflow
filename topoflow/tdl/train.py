"""Train and evaluate: does the topology of the wind field help predict where stagnation persists tomorrow?

    python -m topoflow.tdl.train [--epochs 6] [--level 5]

Chronological split over processed months (no shuffling across time): first 60% train, next 20% validation,
last 20% test. Every model is scored on the same test steps.

  persistence     score = a persistent (>= 24 h) stagnation point within 200 km now
  climatology     score = how often each vertex was a target in the training months
  logistic        logistic regression on the vertex inputs plus lifted wind divergence and speed
  vertex-gnn      the vertex graph network alone (ablation: no simplicial edge branch)
  toponet         simplicial edge convolutions (Hodge L1 up/down) lifted to the vertex network

Metrics: ROC-AUC, average precision (the honest one at a 6% base rate), Brier score, and skill against persistence.
Writes data/tdl/results.json and per-vertex test predictions for the globe (data/tdl/predictions.npz).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ..contract import now_iso
from .complex import icosphere
from .dataset import EDGE_FEATURES, VERTEX_FEATURES, build_cache
from .model import Operators, TopoNet, VertexOnly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "tdl"
MONTHS = ROOT / "data" / "era5" / "months" / "1deg"
PERSIST_COL = VERTEX_FEATURES.index("persistent_now")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "avg_precision": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 5)}


def load(files):
    d = [np.load(f) for f in files]
    return (np.concatenate([x["X_e"] for x in d]).astype("float32"), np.concatenate([x["X_v"] for x in d]).astype("float32"),
            np.concatenate([x["y"] for x in d]), np.concatenate([x["t"] for x in d]))


def lifted(ops: Operators, Xe: np.ndarray) -> np.ndarray:
    """Per vertex: net outflow of the raw wind cochain (signed lift = discrete divergence) and mean edge speed."""
    out = []
    for k in range(len(Xe)):
        x = torch.tensor(Xe[k])
        out.append(torch.cat([torch.sparse.mm(ops.B1s, x[:, :1]), torch.sparse.mm(ops.B1a, x[:, 1:2])], 1).numpy())
    return np.array(out, "float32")


def train_net(net, ops, Xe, Xv, Y, Xe_va, Xv_va, Y_va, epochs, device, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_w = torch.tensor(float((1 - Y.mean()) / max(Y.mean(), 1e-6)), dtype=torch.float32, device=device).clamp(max=20.0)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state = -1.0, None
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        t0, total = time.monotonic(), 0.0
        for k in rng.permutation(len(Xe)):
            xe = torch.tensor(Xe[k], device=device)
            xv = torch.tensor(Xv[k], device=device)
            y = torch.tensor(Y[k], dtype=torch.float32, device=device)
            loss = lossf(net(xe, xv, ops), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        p = predict(net, ops, Xe_va, Xv_va, device)
        ap = average_precision_score(Y_va.ravel(), p.ravel())
        print(f"    epoch {ep + 1}: loss {total / len(Xe):.4f}, val AP {ap:.4f}, {time.monotonic() - t0:.0f}s", flush=True)
        if ap > best:
            best, best_state = ap, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


@torch.no_grad()
def predict(net, ops, Xe, Xv, device) -> np.ndarray:
    net.eval()
    return np.array([torch.sigmoid(net(torch.tensor(Xe[k], device=device), torch.tensor(Xv[k], device=device), ops)).cpu().numpy()
                     for k in range(len(Xe))])


def calibrate(p_val, y_val, p_test):
    """Platt scaling fitted on validation, so Brier scores compare calibrated probabilities."""
    lr = LogisticRegression().fit(np.log(np.clip(p_val, 1e-6, 1 - 1e-6) / (1 - np.clip(p_val, 1e-6, 1 - 1e-6))).reshape(-1, 1)[::7],
                                  y_val.ravel()[::7])
    z = np.log(np.clip(p_test, 1e-6, 1 - 1e-6) / (1 - np.clip(p_test, 1e-6, 1 - 1e-6)))
    return lr.predict_proba(z.reshape(-1, 1))[:, 1].reshape(p_test.shape)


def main():
    ap = argparse.ArgumentParser(prog="topoflow.tdl.train")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--max-months", type=int, default=0, help="use at most this many months (newest dropped first)")
    a = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cx = icosphere(a.level)
    periods = sorted(d.name for d in MONTHS.glob("*") if (d / "summary.json").exists())
    if a.max_months:
        periods = periods[: a.max_months]
    if len(periods) < 5:
        raise SystemExit(f"need at least 5 processed months, have {len(periods)}")
    files = build_cache(cx, MONTHS, periods)
    n = len(periods)
    n_tr, n_va = max(1, int(n * 0.6)), max(1, int(n * 0.2))
    split = {"train": periods[:n_tr], "val": periods[n_tr:n_tr + n_va], "test": periods[n_tr + n_va:]}
    print({k: f"{v[0]}..{v[-1]} ({len(v)})" for k, v in split.items()}, flush=True)
    Xe_tr, Xv_tr, Y_tr, _ = load(files[:n_tr])
    Xe_va, Xv_va, Y_va, _ = load(files[n_tr:n_tr + n_va])
    Xe_te, Xv_te, Y_te, T_te = load(files[n_tr + n_va:])

    # Standardise with training statistics only. Flow keeps its sign: scale, no shift.
    mu_v, sd_v = Xv_tr.mean((0, 1)), Xv_tr.std((0, 1)) + 1e-6
    sd_e = Xe_tr.std((0, 1)) + 1e-6
    norm_v = lambda x: (x - mu_v) / sd_v  # noqa: E731
    norm_e = lambda x: x / sd_e  # noqa: E731

    results: dict[str, dict] = {}
    y = Y_te.ravel()
    p_persist = Xv_te[..., PERSIST_COL].ravel()
    results["persistence"] = metrics(y, p_persist)
    clim = Y_tr.mean(0)
    results["climatology"] = metrics(y, np.broadcast_to(clim, Y_te.shape).ravel())

    ops_cpu = Operators(cx, "cpu")
    L_tr, L_te, L_va = lifted(ops_cpu, Xe_tr), lifted(ops_cpu, Xe_te), lifted(ops_cpu, Xe_va)
    sub = slice(None, None, 9)  # every 9th vertex-step keeps the logistic fit fast; it is linear and data-rich
    feat = lambda Xv, L: np.concatenate([Xv, L], -1).reshape(-1, Xv.shape[-1] + 2)  # noqa: E731
    F_tr, F_te = feat(Xv_tr, L_tr), feat(Xv_te, L_te)
    mu_f, sd_f = F_tr.mean(0), F_tr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(((F_tr - mu_f) / sd_f)[sub], Y_tr.ravel()[sub])
    p_lr_va = lr.predict_proba((feat(Xv_va, L_va) - mu_f) / sd_f)[:, 1].reshape(Y_va.shape)
    p_lr = lr.predict_proba((F_te - mu_f) / sd_f)[:, 1].reshape(Y_te.shape)
    results["logistic"] = metrics(y, calibrate(p_lr_va, Y_va, p_lr).ravel())

    ops = Operators(cx, device)
    preds = {}
    for name, net in (("vertex-gnn", VertexOnly(len(VERTEX_FEATURES))),
                      ("toponet", TopoNet(len(EDGE_FEATURES), len(VERTEX_FEATURES)))):
        print(f"  training {name} on {device}", flush=True)
        net = train_net(net.to(device), ops, norm_e(Xe_tr), norm_v(Xv_tr), Y_tr, norm_e(Xe_va), norm_v(Xv_va), Y_va,
                        a.epochs, device)
        p_va = predict(net, ops, norm_e(Xe_va), norm_v(Xv_va), device)
        p_te = calibrate(p_va, Y_va, predict(net, ops, norm_e(Xe_te), norm_v(Xv_te), device))
        results[name] = metrics(y, p_te.ravel())
        preds[name] = p_te
        if name == "toponet":
            torch.save(net.state_dict(), OUT / "toponet.pt")

    base = results["persistence"]["avg_precision"]
    for r in results.values():
        r["ap_gain_over_persistence"] = round(r["avg_precision"] - base, 4)
    report = {
        "created_at": now_iso(), "split": split, "test_steps": int(len(Y_te)), "vertices": int(Y_te.shape[1]),
        "base_rate": round(float(y.mean()), 4), "target": "a stagnation point persisting >= 24 h within 200 km, 24 h ahead",
        "mesh": f"icosphere level {a.level}", "epochs": a.epochs, "device": str(device), "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / "predictions.npz", lat=cx.lat.astype("float32"), lon=cx.lon.astype("float32"),
                        t=T_te, toponet=preds["toponet"].astype("float16"), truth=Y_te,
                        persistence=Xv_te[..., PERSIST_COL].astype(bool))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
