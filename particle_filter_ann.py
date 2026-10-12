"""
particle_filter_ann.py  --  Pipeline 1, step 2: particle filter informed by the ANN
====================================================================================
Implements Sec. IV-E of the paper, at every time step:
 1. Initialisation: N particles = ANN classified pose + delta sampled from the
    (true - classified) validation-set error distribution.
 2. Propagation through the nonlinear dynamics.  The leader's forward speed is unknown ->
    sampled uniformly (disturbance); its attitude (which tilts its velocity vector) follows a
    random walk.  Follower controls are known.
 3. Each particle renders synthetic feature pixel locations; weight = 1 / MSE against the
    detected features (undetected = occluded features are ignored; occlusion is NOT
    ray-traced per particle, as in the paper).
 4. Resample floor(alpha*N) particles by weight; the remaining N - floor(alpha*N) are drawn from
    "ANN classified pose + error distribution".

Two paper configurations (Fig. 9, Fig. 10):
   A:  N = 1000, alpha = 0.9 every 10 cycles (alpha = 1 otherwise)
   B:  N = 5000, alpha = 0.9 every cycle
Outputs: results/pf_{A,B}.npz and figures/fig09_*.png, fig10_*.png
(grey dashed line = ANN label alone, for reference; not in the paper).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import dataset_generation as dg

PF_CONFIGS = {
    "A": dict(n_particles=1000, alpha=0.9, inject_every=10, fig="fig09_pf_N1000_alpha0.9_every10cycles.png"),
    "B": dict(n_particles=5000, alpha=0.9, inject_every=1, fig="fig10_pf_N5000_alpha0.9_everycycle.png"),
}


def systematic_resample(w, m, rng):
    cs = np.cumsum(w)
    cs[-1] = 1.0
    return np.searchsorted(cs, (rng.random() + np.arange(m)) / m)


class ParticleFilter:
    def __init__(self, n_particles, delta, rng, alpha=0.9, inject_every=1,
                 v_leader_range=(20.0, 30.0), att_sigma=np.deg2rad([1.0, 0.5, 0.3]),
                 pitch_yaw_prior=np.deg2rad(3.0), eps=1e-3):
        self.N, self.delta, self.rng = n_particles, delta, rng
        self.alpha, self.inject_every = alpha, inject_every
        self.v_range = v_leader_range
        self.att_sigma, self.py_prior, self.eps = att_sigma, pitch_yaw_prior, eps
        self.states = None
        self.w = np.full(n_particles, 1.0 / n_particles)

    def _sample_classifier(self, cls_pose4, n):
        """ANN classified pose + error distribution (pitch/yaw are not in the label -> prior)."""
        d = self.delta[self.rng.integers(0, len(self.delta), n)]
        pose4 = cls_pose4[None] + d
        py = self.rng.normal(0.0, self.py_prior, (n, 2))
        return np.column_stack([pose4, py])

    def initialize(self, cls_pose4):
        self.states = self._sample_classifier(cls_pose4, self.N)

    def predict(self, uF):
        n, r = self.N, self.rng
        vL = r.uniform(*self.v_range, n)          # unknown leader speed -> uniform disturbance
        p = dg.propagate_position(self.states, uF, vL)
        att = self.states[:, 3:6] + r.normal(0.0, 1.0, (n, 3)) * self.att_sigma
        self.states = np.column_stack([p, att])

    def update(self, uv_meas, det):
        """Weight update; returns the weighted-mean state estimate."""
        if det.sum() == 0:
            w = np.full(self.N, 1.0 / self.N)
        else:
            uv_p = dg.project(self.states)                              # (N,14,2) synthetic features
            diff = uv_p[:, det, :] - uv_meas[None, det, :]
            mse = np.mean(np.sum(diff ** 2, axis=2), axis=1)
            w = 1.0 / (mse + self.eps)
            w /= w.sum()
        self.w = w
        return w @ self.states

    def resample(self, k, cls_pose4):
        alpha = self.alpha if (k % self.inject_every == 0) else 1.0
        m = int(np.floor(alpha * self.N))
        new = self.states[systematic_resample(self.w, m, self.rng)]
        if m < self.N:
            new = np.vstack([new, self._sample_classifier(cls_pose4, self.N - m)])
        self.states = new


def run_particle_filter(traj, ann_model, delta, n_particles, alpha, inject_every, seed=0):
    from train_ann import predict_labels      # local import: keeps torch optional elsewhere
    rng = np.random.default_rng(seed)
    K = len(traj["times"])
    X = dg.encode_features(traj["uv"], traj["detected"])
    cls_pose = dg.label_to_pose(predict_labels(ann_model, X))             # ANN every cycle
    pf = ParticleFilter(n_particles, delta, rng, alpha, inject_every)
    pf.initialize(cls_pose[0])
    est = np.zeros((K, 6))
    t0 = time.perf_counter()
    for k in range(K):
        if k > 0:
            pf.predict(traj["uF"][k - 1])
        est[k] = pf.update(traj["uv"][k], traj["detected"][k])
        pf.resample(k, cls_pose[k])
    ms = 1000 * (time.perf_counter() - t0) / K
    truth = traj["states"]
    base = np.column_stack([cls_pose, np.zeros((K, 2))])
    return dict(times=traj["times"], est=est,
                pos_err=dg.position_error(est, truth), att_err=dg.attitude_error_deg(est, truth),
                base_pos_err=dg.position_error(base, truth), base_att_err=dg.attitude_error_deg(base, truth),
                runtime_ms_per_frame=ms, n_particles=n_particles, alpha=alpha, inject_every=inject_every)


def pf_title(c):
    every = "every cycle" if c["inject_every"] == 1 else f"every {c['inject_every']} cycles"
    return f"N = {c['n_particles']}, alpha = {c['alpha']} {every}"


def run(out_dir="outputs", seed=0, configs=("A", "B")):
    from train_ann import load_ann
    dirs = dg.get_dirs(out_dir)
    traj = dg.load_trajectory(out_dir)
    ann = load_ann(out_dir)
    delta = np.load(dirs["data"] / "ann_error_dist.npz")["delta"]
    results = {}
    for name in configs:
        c = PF_CONFIGS[name]
        res = run_particle_filter(traj, ann, delta, c["n_particles"], c["alpha"], c["inject_every"], seed)
        dg.save_result(dirs["results"] / f"pf_{name}.npz", res)
        dg.plot_error_traces(res["times"], res["pos_err"], res["att_err"], dirs["figures"] / c["fig"],
                             title=pf_title(c), baseline=(res["base_pos_err"], res["base_att_err"]),
                             baseline_label="ANN label only")
        print(f"[pf {name}] {pf_title(c)}: pos err mean {res['pos_err'].mean():.2f} m "
              f"(max {res['pos_err'].max():.2f}), att err mean {res['att_err'].mean():.2f} deg, "
              f"{res['runtime_ms_per_frame']:.1f} ms/frame | ANN label only: "
              f"{res['base_pos_err'].mean():.2f} m, {res['base_att_err'].mean():.2f} deg")
        results[name] = res
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=["A", "B"], choices=list(PF_CONFIGS))
    a = ap.parse_args()
    run(a.out, a.seed, a.configs)
