"""
bundle_adjustment_xgb.py  --  Pipeline 2, step 2: bundle adjustment on XGBoost output
=====================================================================================
XGBoost regresses (x, y, z, phi) from the detected features.  Bundle adjustment then
refines the full relative pose s = [x, y, z, phi, theta, psi] by minimising, with
scipy.optimize.least_squares (Levenberg-Marquardt-style trust region), the cost

   sum_frames sum_detected-features || pi(s_k, X_j) - u_kj ||^2 / sigma_px^2     (reprojection)
 + sum_frames || (s_k - s_k^prior) / sigma_prior ||^2                             (XGBoost prior)
 + sum_frames || (s_{k+1} - 2 s_k + s_{k-1}) / sigma_smooth ||^2                   (window mode only)

* pi(.) is the same pinhole/gimbal camera model used to generate the data and X_j are the
  known 3-D feature positions on the aircraft.
* Prior mean = XGBoost (x,y,z,phi) and 0 for pitch/yaw.  Prior std-devs = XGBoost validation
  RMSE (so the prior is exactly as trusted as it deserves) and 3 deg for pitch/yaw.  The prior
  matters because pitch / yaw are only weakly observable and are confounded with position.
* window = 1  -> single-frame refinement ("PnP-style" BA), initialised from XGBoost.
  window = K  -> sliding-window BA over the last K frames (true multi-view BA with a
                 constant-velocity smoothness term), initialised by tracking.

Outputs: results/ba_single.npz, results/ba_window.npz, results/ba_static.json,
         figures/figXC_ba_single_*.png, figXD_ba_window_*.png, figXE_static_error_cdf.png
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset_generation as dg


@dataclass
class BAConfig:
    sigma_px: float = 1.0
    prior_scale: float = 1.0
    att_prior_sigma_deg: float = 3.0
    smooth_pos_sigma: float = 1.0       # m
    smooth_att_sigma_deg: float = 2.0
    window: int = 1
    max_nfev: int = 60
    loss: str = "linear"                # or "soft_l1" for robustness to outlier detections


def make_prior_sigma(xgb_val_rmse, cfg):
    s = np.maximum(np.asarray(xgb_val_rmse, float) * cfg.prior_scale, [0.5, 0.5, 0.5, np.deg2rad(1.0)])
    a = np.deg2rad(cfg.att_prior_sigma_deg)
    return np.concatenate([s, [a, a]])


def _residuals(theta, uv, det, prior_mean, prior_sigma, cfg):
    K = uv.shape[0]
    S = theta.reshape(K, 6)
    proj = dg.project(S)
    r_px = np.where(det[..., None], (proj - np.nan_to_num(uv)) / cfg.sigma_px, 0.0).ravel()
    parts = [r_px, ((S - prior_mean) / prior_sigma).ravel()]
    if K >= 3:
        acc = S[2:] - 2 * S[1:-1] + S[:-2]
        sc = np.array([cfg.smooth_pos_sigma] * 3 + [np.deg2rad(cfg.smooth_att_sigma_deg)] * 3)
        parts.append((acc / sc).ravel())
    return np.concatenate(parts)


def solve(uv, det, prior_mean, init, prior_sigma, cfg):
    """uv (K,14,2), det (K,14), prior_mean/init (K,6) -> refined states (K,6)."""
    sol = least_squares(_residuals, init.ravel(), args=(uv, det, prior_mean, prior_sigma, cfg),
                        method="trf", x_scale="jac", loss=cfg.loss, f_scale=3.0, max_nfev=cfg.max_nfev)
    return sol.x.reshape(-1, 6)


def run_ba_trajectory(traj, pred4, prior_sigma, cfg, init_mode):
    """init_mode: 'xgb' (every frame starts from the XGBoost estimate) or 'track'."""
    K = len(traj["times"])
    prior = np.column_stack([pred4, np.zeros((K, 2))])
    hist, causal, t0 = prior.copy(), prior.copy(), time.perf_counter()
    for k in range(K):
        lo = max(0, k - cfg.window + 1)
        init = prior[lo:k + 1].copy()
        if init_mode == "track" and k > 0:
            init[:-1] = hist[lo:k]
            init[-1] = hist[k - 1]
        sol = solve(traj["uv"][lo:k + 1], traj["detected"][lo:k + 1], prior[lo:k + 1], init, prior_sigma, cfg)
        hist[lo:k + 1] = sol
        causal[k] = sol[-1]
    ms = 1000 * (time.perf_counter() - t0) / K
    truth = traj["states"]
    base = prior
    return dict(times=traj["times"], est=causal,
                pos_err=dg.position_error(causal, truth), att_err=dg.attitude_error_deg(causal, truth),
                base_pos_err=dg.position_error(base, truth), base_att_err=dg.attitude_error_deg(base, truth),
                runtime_ms_per_frame=ms, window=cfg.window)


def evaluate_static(data, pred_fn, prior_sigma, cfg, n_eval=1000, seed=0):
    """Independent single-frame evaluation on test-set samples (no temporal information)."""
    rng = np.random.default_rng(seed)
    te = np.where(data["split"] == 2)[0]
    idx = rng.choice(te, min(n_eval, len(te)), replace=False)
    X = dg.encode_features(data["uv"][idx], data["detected"][idx])
    pred4 = pred_fn(X)
    truth = data["states"][idx]
    prior = np.column_stack([pred4, np.zeros((len(idx), 2))])
    est = np.zeros_like(prior)
    t0 = time.perf_counter()
    for i, j in enumerate(idx):
        est[i] = solve(data["uv"][j][None], data["detected"][j][None], prior[i:i + 1], prior[i:i + 1],
                       prior_sigma, cfg)[0]
    ms = 1000 * (time.perf_counter() - t0) / len(idx)
    out = dict(n_eval=len(idx), runtime_ms_per_sample=ms)
    for name, e in (("xgb_only", prior), ("xgb_plus_ba", est)):
        pe, ae = dg.position_error(e, truth), dg.attitude_error_deg(e, truth)
        out[name] = dict(pos_rmse_m=float(np.sqrt(np.mean(pe ** 2))), pos_median_m=float(np.median(pe)),
                         pos_p95_m=float(np.percentile(pe, 95)), att_mean_deg=float(ae.mean()),
                         att_median_deg=float(np.median(ae)), att_p95_deg=float(np.percentile(ae, 95)))
    out["_errs"] = dict(pos_xgb=dg.position_error(prior, truth), pos_ba=dg.position_error(est, truth),
                        att_xgb=dg.attitude_error_deg(prior, truth), att_ba=dg.attitude_error_deg(est, truth))
    return out


def fig_error_cdf(errs, path):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for a, key, lab in ((ax[0], "pos", "position error [m]"), (ax[1], "att", "attitude error [deg]")):
        for suffix, nm in (("xgb", "XGBoost only"), ("ba", "XGBoost + BA")):
            e = np.sort(errs[f"{key}_{suffix}"])
            a.plot(e, np.arange(1, len(e) + 1) / len(e), label=nm)
        a.set_xlabel(lab); a.set_ylabel("CDF"); a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(out_dir="outputs", window=5, n_eval=1000, prior_scale=1.0, loss="linear"):
    from train_xgboost import load_xgb, predict_xgb      # local import keeps xgboost optional here
    dirs = dg.get_dirs(out_dir)
    traj, data = dg.load_trajectory(out_dir), dg.load_dataset(out_dir)
    models = load_xgb(out_dir)
    val_rmse = json.loads((dirs["results"] / "xgb_metrics.json").read_text())["val_rmse"]
    pred_fn = lambda X: predict_xgb(models, X)
    pred4 = pred_fn(dg.encode_features(traj["uv"], traj["detected"]))

    cfg1 = BAConfig(window=1, prior_scale=prior_scale, loss=loss)
    cfgK = BAConfig(window=window, prior_scale=prior_scale, loss=loss)
    sig = make_prior_sigma(val_rmse, cfg1)
    results = {}
    for name, cfg, mode, fig in (("single", cfg1, "xgb", "figXC_ba_single_frame.png"),
                                 ("window", cfgK, "track", f"figXD_ba_window{window}.png")):
        res = run_ba_trajectory(traj, pred4, sig, cfg, mode)
        dg.save_result(dirs["results"] / f"ba_{name}.npz", res)
        ttl = "XGBoost + BA (single frame)" if name == "single" else f"XGBoost + BA (sliding window K={window})"
        dg.plot_error_traces(res["times"], res["pos_err"], res["att_err"], dirs["figures"] / fig,
                             title=ttl, baseline=(res["base_pos_err"], res["base_att_err"]),
                             baseline_label="XGBoost only")
        print(f"[ba {name}] pos err mean {res['pos_err'].mean():.2f} m (max {res['pos_err'].max():.2f}), "
              f"att err mean {res['att_err'].mean():.2f} deg, {res['runtime_ms_per_frame']:.1f} ms/frame | "
              f"XGBoost only: {res['base_pos_err'].mean():.2f} m, {res['base_att_err'].mean():.2f} deg")
        results[name] = res

    st = evaluate_static(data, pred_fn, sig, cfg1, n_eval)
    errs = st.pop("_errs")
    fig_error_cdf(errs, dirs["figures"] / "figXE_static_error_cdf.png")
    (dirs["results"] / "ba_static.json").write_text(json.dumps(dg.to_jsonable(st), indent=2))
    print("[ba static]", json.dumps(dg.to_jsonable(st), indent=2))
    return results, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--prior-scale", type=float, default=1.0)
    ap.add_argument("--loss", default="linear", choices=["linear", "soft_l1", "huber"])
    a = ap.parse_args()
    run(a.out, a.window, a.n_eval, a.prior_scale, a.loss)