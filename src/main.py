"""
run_pipeline.py  --  runs everything end to end
===============================================
  dataset      : dataset_generation.py   (dataset + trajectory, Figs 2, 3, 6, 7, 8)
  ann          : train_ann.py            (Pipeline 1: ANN,            Figs 4, 5)
  pf           : particle_filter_ann.py  (Pipeline 1: particle filter, Figs 9, 10)
  xgb          : train_xgboost.py        (Pipeline 2: XGBoost regression)
  ba           : bundle_adjustment_xgb.py(Pipeline 2: bundle adjustment)
  report       : Fig. 1 architecture diagrams, comparison figure, "Reported vs Yours" tables

Examples
  python main.py                               # everything, defaults
  python main.py --stages report               # regenerate tables / comparison figure only
  python main.py --stages ann pf --epochs 60   # retrain ANN + rerun PF
  python main.py --n-samples 50000 --epochs 5 --n-estimators 100   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import dataset_generation as dg

ALL_STAGES = ["dataset", "ann", "pf", "xgb", "ba", "report"]

# Values reported in the paper (PF numbers are read approximately off Figs. 9-10).
REPORTED = dict(
    n_labels=4800, ann_val_acc=0.54,
    pf_A=dict(pos="~5 - 19 m", att="~0 - 3 deg"),
    pf_B=dict(pos="~3 - 34 m (mostly 5-15 m)", att="~0 - 3 deg"),
)


# --------------------------------------------------------------------------- #
# Fig. 1 architecture diagrams
# --------------------------------------------------------------------------- #
def _box(ax, xy, w, h, text, color="#0b5ea8"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=color, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, color="white", ha="center", va="center",
            fontsize=8, fontweight="bold", wrap=True)


def _arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.5, color="k"))


def fig_architecture(path, pipeline=1):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    _box(ax, (2.5, 6.4), 5, 1.2, "Convolutional neural network\nfeature detector (assumed given)")
    if pipeline == 1:
        _box(ax, (0.8, 3.4), 4, 1.1, "Neural network coarse\npose classifier (ANN)")
        _box(ax, (4.8, 0.8), 4.2, 1.1, "Relative pose particle filter")
        _arrow(ax, (4.0, 6.4), (2.8, 4.5)); _arrow(ax, (7.0, 6.4), (7.6, 1.9))
        _arrow(ax, (3.5, 3.4), (5.8, 1.9))
        ax.text(7.6, 4.2, "feature pixel\nlocations", fontsize=7, ha="center")
        ax.add_patch(Rectangle((0.3, 0.4), 9.4, 4.5, fill=False, ls=":", ec="red", lw=1.5))
        ax.text(0.5, 0.55, "Pipeline 1 (paper)", color="red", fontsize=8)
    else:
        _box(ax, (0.8, 3.4), 4, 1.1, "XGBoost regressor\n(x, y, z, roll)", "#2a7f3f")
        _box(ax, (4.8, 0.8), 4.2, 1.1, "Bundle adjustment\n(reprojection + prior)", "#2a7f3f")
        _arrow(ax, (4.0, 6.4), (2.8, 4.5)); _arrow(ax, (7.0, 6.4), (7.6, 1.9))
        _arrow(ax, (3.5, 3.4), (5.8, 1.9))
        ax.text(7.6, 4.2, "feature pixel\nlocations", fontsize=7, ha="center")
        ax.add_patch(Rectangle((0.3, 0.4), 9.4, 4.5, fill=False, ls=":", ec="green", lw=1.5))
        ax.text(0.5, 0.55, "Pipeline 2 (proposed)", color="green", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _stats(r):
    return dict(pos_mean=float(r["pos_err"].mean()), pos_max=float(r["pos_err"].max()),
                pos_rmse=float(np.sqrt(np.mean(r["pos_err"] ** 2))),
                att_mean=float(r["att_err"].mean()), att_max=float(r["att_err"].max()),
                ms=float(r["runtime_ms_per_frame"]))


def report(out_dir):
    dirs = dg.get_dirs(out_dir)
    fd, rd = dirs["figures"], dirs["results"]
    fig_architecture(fd / "fig01_architecture_pipeline1.png", 1)
    fig_architecture(fd / "figXF_architecture_pipeline2.png", 2)

    def jload(n):
        p = rd / n
        return json.loads(p.read_text()) if p.exists() else None

    def rload(n):
        p = rd / n
        return dg.load_result(p) if p.exists() else None

    ann, xgbm, static = jload("ann_metrics.json"), jload("xgb_metrics.json"), jload("ba_static.json")
    runs = {"ANN label only": None, "PF A (N=1000, a=0.9/10 cyc)": rload("pf_A.npz"),
            "PF B (N=5000, a=0.9/cyc)": rload("pf_B.npz"), "XGBoost only": None,
            "XGBoost + BA (single frame)": rload("ba_single.npz"),
            "XGBoost + BA (window)": rload("ba_window.npz")}
    # baselines (ANN label / XGB only) are stored inside the filter results
    for key, src in (("ANN label only", "PF A (N=1000, a=0.9/10 cyc)"), ("XGBoost only", "XGBoost + BA (single frame)")):
        if runs[src] is not None:
            r = runs[src]
            runs[key] = dict(times=r["times"], pos_err=r["base_pos_err"], att_err=r["base_att_err"],
                             runtime_ms_per_frame=0.0)

    L = ["# Results summary", ""]
    L += ["## Part 1: reproduction (Reported vs. Yours)", "",
          "| Quantity | Reported (paper) | Ours |", "|---|---|---|",
          f"| Number of pose labels | {REPORTED['n_labels']} | {dg.N_LABELS} |"]
    if ann:
        L.append(f"| ANN validation accuracy | {REPORTED['ann_val_acc']:.0%} | {ann['val_accuracy']:.1%} |")
        L.append(f"| ANN val. accuracy, correct or adjacent cell | 'incorrect labels are commonly close' | "
                 f"{ann['val_adjacent_cell_accuracy']:.1%} |")
    for k, nm in (("PF A (N=1000, a=0.9/10 cyc)", "pf_A"), ("PF B (N=5000, a=0.9/cyc)", "pf_B")):
        r = runs[k]
        if r is not None:
            s = _stats(r)
            L.append(f"| {k}: position error | {REPORTED[nm]['pos']} | {s['pos_mean']:.1f} m mean "
                     f"(max {s['pos_max']:.1f}) |")
            L.append(f"| {k}: attitude error | {REPORTED[nm]['att']} | {s['att_mean']:.1f} deg mean "
                     f"(max {s['att_max']:.1f}) |")
    L += ["", "## Part 2: extension (XGBoost + bundle adjustment) on the 10 s approach trajectory", "",
          "| Method | Pos. mean [m] | Pos. RMSE [m] | Pos. max [m] | Att. mean [deg] | Att. max [deg] | ms/frame |",
          "|---|---|---|---|---|---|---|"]
    for k, r in runs.items():
        if r is not None:
            s = _stats(r)
            L.append(f"| {k} | {s['pos_mean']:.2f} | {s['pos_rmse']:.2f} | {s['pos_max']:.2f} | "
                     f"{s['att_mean']:.2f} | {s['att_max']:.2f} | {s['ms']:.1f} |")
    if ann and xgbm and static:
        L += ["", "## Static (independent single-frame) test-set comparison", "",
              "| Method | Pos. RMSE [m] | Pos. median [m] | Att. mean [deg] |", "|---|---|---|---|",
              f"| ANN label only (val set) | {ann['val_pos_rmse_3d_m']:.2f} | - | - |",
              f"| XGBoost only | {static['xgb_only']['pos_rmse_m']:.2f} | {static['xgb_only']['pos_median_m']:.2f} | "
              f"{static['xgb_only']['att_mean_deg']:.2f} |",
              f"| XGBoost + BA | {static['xgb_plus_ba']['pos_rmse_m']:.2f} | {static['xgb_plus_ba']['pos_median_m']:.2f} | "
              f"{static['xgb_plus_ba']['att_mean_deg']:.2f} |"]
    (rd / "results_summary.md").write_text("\n".join(L))
    print("\n".join(L))

    # extra figure: all methods on the trajectory
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    for k, r in runs.items():
        if r is None:
            continue
        ls = "--" if "only" in k else "-"
        ax[0].plot(r["times"], r["pos_err"], ls, lw=1.1, label=k)
        ax[1].plot(r["times"], r["att_err"], ls, lw=1.1, label=k)
    ax[0].set_ylabel("position error [m]"); ax[1].set_ylabel("attitude error [deg]")
    ax[1].set_xlabel("time [s]"); ax[0].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.3); a.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(fd / "figXG_compare_all_methods.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1_000_000)
    ap.add_argument("--att-jitter-deg", type=float, default=0.0, help="random leader pitch/yaw in the dataset [deg]")
    ap.add_argument("--min-range", type=float, default=None, help="closest sampled follower-leader distance [m]")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--window", type=int, default=5)
    a = ap.parse_args()

    for st in ALL_STAGES:
        if st not in a.stages:
            continue
        t0 = time.time()
        print(f"\n===== stage: {st} =====")
        if st == "dataset":
            dg.generate_all(a.out, a.n_samples, a.seed, att_jitter_deg=a.att_jitter_deg, min_range=a.min_range)
        elif st == "ann":
            import train_ann
            train_ann.run(a.out, epochs=a.epochs, seed=a.seed)
        elif st == "pf":
            import particle_filter_ann
            particle_filter_ann.run(a.out, a.seed)
        elif st == "xgb":
            import train_xgboost
            train_xgboost.run(a.out, n_estimators=a.n_estimators, seed=a.seed)
        elif st == "ba":
            import bundle_adjustment_xgb
            bundle_adjustment_xgb.run(a.out, window=a.window)
        elif st == "report":
            report(a.out)
        print(f"----- {st} done in {time.time() - t0:.1f}s")
    print(f"\nAll outputs are in ./{a.out}/ (data, models, figures, results)")


if __name__ == "__main__":
    main()