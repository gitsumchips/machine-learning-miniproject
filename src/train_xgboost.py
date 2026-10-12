"""
train_xgboost.py  --  Pipeline 2, step 1: XGBoost regression of (x, y, z, phi)
==============================================================================
Same inputs (42 features), same train/val/test split and same ground truth as the ANN,
but instead of classifying into 4800 cells, four XGBoost regressors directly predict the
continuous x, y, z [m] and roll phi [rad].  No discretisation error.

Why XGBoost: tabular, low-dimensional inputs with missing-feature flags are a natural
fit for gradient-boosted trees; they are cheap to train, need little tuning, and output
continuous values that can then be refined by bundle adjustment.

Outputs: models/xgb_{x,y,z,phi}.json, results/xgb_metrics.json (val RMSE is reused as the
prior std-dev inside bundle adjustment), figures/figXA_xgb_error_vs_true_pose.png,
figures/figXB_xgb_pred_vs_true.png
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset_generation as dg

TARGETS = ["x", "y", "z", "phi"]


def load_xgb(out_dir="outputs"):
    d = dg.get_dirs(out_dir)
    models = []
    for n in TARGETS:
        m = xgb.XGBRegressor()
        m.load_model(str(d["models"] / f"xgb_{n}.json"))
        models.append(m)
    return models


def predict_xgb(models, X):
    return np.column_stack([m.predict(X) for m in models])


def run(out_dir="outputs", n_estimators=800, lr=0.05, max_depth=8, early_stop=40, seed=0):
    dirs = dg.get_dirs(out_dir)
    data = dg.load_dataset(out_dir)
    X = dg.encode_features(data["uv"], data["detected"])
    Y = data["states"][:, :4]
    split = data["split"]
    tr, va, te = (np.where(split == s)[0] for s in (0, 1, 2))

    for j, name in enumerate(TARGETS):
        m = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             n_jobs=-1, random_state=seed, eval_metric="rmse",
                             early_stopping_rounds=early_stop)
        m.fit(X[tr], Y[tr, j], eval_set=[(X[va], Y[va, j])], verbose=False)
        m.save_model(str(dirs["models"] / f"xgb_{name}.json"))
        print(f"[xgb] {name}: best iteration {m.best_iteration}")

    models = load_xgb(out_dir)
    pv, pt = predict_xgb(models, X[va]), predict_xgb(models, X[te])
    ev, et = pv - Y[va], pt - Y[te]
    rv, rt = np.sqrt(np.mean(ev ** 2, 0)), np.sqrt(np.mean(et ** 2, 0))

    dg.plot_error_hists(ev, dirs["figures"] / "figXA_xgb_error_vs_true_pose.png")

    sub = np.random.default_rng(seed).choice(len(te), min(3000, len(te)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    for j, ax in enumerate(axes.ravel()):
        sc = 180 / np.pi if j == 3 else 1.0
        ax.scatter(Y[te][sub, j] * sc, pt[sub, j] * sc, s=2, alpha=0.4)
        lim = [Y[te][:, j].min() * sc, Y[te][:, j].max() * sc]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("true " + dg.POSE4_NAMES[j]); ax.set_ylabel("predicted"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(dirs["figures"] / "figXB_xgb_pred_vs_true.png", dpi=150); plt.close(fig)

    metrics = dict(val_rmse=rv, test_rmse=rt,       # order x, y, z [m], phi [rad]
                   val_pose_rmse=dict(x_m=rv[0], y_m=rv[1], z_m=rv[2], phi_deg=np.degrees(rv[3])),
                   test_pose_rmse=dict(x_m=rt[0], y_m=rt[1], z_m=rt[2], phi_deg=np.degrees(rt[3])),
                   test_pos_rmse_3d_m=float(np.sqrt(np.mean(np.sum(et[:, :3] ** 2, 1)))),
                   n_estimators=n_estimators, lr=lr, max_depth=max_depth)
    (dirs["results"] / "xgb_metrics.json").write_text(json.dumps(dg.to_jsonable(metrics), indent=2))
    print("[xgb]", json.dumps(dg.to_jsonable(metrics), indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.out, a.n_estimators, a.lr, a.max_depth, seed=a.seed)