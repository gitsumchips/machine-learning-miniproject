"""
dataset_generation.py
=====================
Synthetic data for "Vision-Based Precision Pose Estimation for Autonomous Formation Flying",
plus the geometry / dynamics / plotting helpers shared by the other scripts
(they do `import dataset_generation as dg`).

Rendering model (from the earlier generate_dataset.py)
------------------------------------------------------
* Coarse procedural F-16-like TRIANGLE MESH (fuselage rings, wings, stabilisers, fin) with
  14 keypoints that are mesh vertices (body frame: x forward, y right, z down).
* Pinhole camera, defaults: width 640, focal 400, height 480.
* Occlusion by two-sided Moller-Trumbore ray casting from the camera to every keypoint.
* A keypoint that is occluded / behind the camera / outside the image is "not detected"
  (occ = 1).  Network input layout, 14 x [u, v, occ] = 42 numbers.

Conventions (assumptions - the paper does not specify them)
-----------------------------------------------------------
* State s = [x, y, z, phi, theta, psi]
    (x, y, z)           follower position in the heading-aligned level frame centred on the leader
                        (x fwd, y right, z down).  x < 0: follower behind the leader.
    (phi, theta, psi)   leader roll, pitch, yaw relative to the follower's level frame [rad]
                        (R_L = Rz(psi) Ry(theta) Rx(phi)).
* The camera sits on the follower, is gimballed at the leader origin, and keeps its down axis
  as close to world-down as possible (no extra camera roll).
* The label grid (Table I) covers (x, y, z, phi).  Pitch and yaw are ignored by the labels, as in
  the paper; by default they are 0 in the dataset (--att-jitter-deg adds random pitch / yaw).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Aircraft model: coarse F-16-like triangle mesh + 14 keypoints
# --------------------------------------------------------------------------- #

#
class AircraftModel:
    N_FEATURES = 14
    FEATURE_NAMES = ["nose tip", "tail cone", "right wingtip LE", "left wingtip LE",
                     "right wingtip TE", "left wingtip TE", "right stab tip", "left stab tip",
                     "fin top TE", "fin top LE", "canopy top", "belly / intake",
                     "right wing root TE", "left wing root TE"]

    def __init__(self):
        verts, tris = [], []

        def add_v(p):
            verts.append(p)
            return len(verts) - 1

        stations = [(5.0, 0.45, 0.45), (2.0, 0.80, 0.90), (-2.0, 0.90, 0.90),
                    (-6.0, 0.80, 0.80), (-7.5, 0.60, 0.60)]       # (x, semi-axis y, semi-axis z)
        n_side = 8
        ang = np.arange(n_side) * 2 * np.pi / n_side
        nose = add_v((7.5, 0.0, 0.0))
        rings = []
        for x, ry, rz in stations:
            # angle 0 -> +y (right), 90 deg -> +z (down), 270 deg -> top (z negative)
            rings.append([add_v((x, ry * np.cos(a), rz * np.sin(a))) for a in ang])
        tail = add_v((-8.0, 0.0, 0.0))
        for k in range(n_side):
            k2 = (k + 1) % n_side
            tris.append((nose, rings[0][k], rings[0][k2]))
            tris.append((tail, rings[-1][k2], rings[-1][k]))
        for r in range(len(rings) - 1):
            for k in range(n_side):
                k2 = (k + 1) % n_side
                a, b, c, d = rings[r][k], rings[r][k2], rings[r + 1][k2], rings[r + 1][k]
                tris += [(a, b, c), (a, c, d)]

        def add_quad(p0, p1, p2, p3):
            i = [add_v(p) for p in (p0, p1, p2, p3)]
            tris.extend([(i[0], i[1], i[2]), (i[0], i[2], i[3])])
            return i

        wing_r = add_quad((0.5, 0.9, 0), (-5.0, 0.85, 0), (-4.5, 4.8, 0), (-2.5, 4.8, 0))
        wing_l = add_quad((0.5, -0.9, 0), (-5.0, -0.85, 0), (-4.5, -4.8, 0), (-2.5, -4.8, 0))
        stab_r = add_quad((-5.0, 0.8, 0), (-7.0, 0.6, 0), (-7.8, 2.6, 0), (-6.5, 2.6, 0))
        stab_l = add_quad((-5.0, -0.8, 0), (-7.0, -0.6, 0), (-7.8, -2.6, 0), (-6.5, -2.6, 0))
        fin = add_quad((-3.0, 0, -0.9), (-7.0, 0, -0.7), (-7.2, 0, -3.8), (-5.2, 0, -3.8))

        self.verts = np.array(verts, dtype=np.float64)
        self.tris = np.array(tris, dtype=np.int64)
        top, bottom = 6, 2
        kp_idx = [nose, tail, wing_r[3], wing_l[3], wing_r[2], wing_l[2], stab_r[2], stab_l[2],
                  fin[2], fin[3], rings[1][top], rings[1][bottom], wing_r[1], wing_l[1]]
        self.keypoints = self.verts[kp_idx]                 # (14, 3), exactly on the surface
        v0 = self.verts[self.tris[:, 0]]
        self.v0 = v0
        self.e1 = self.verts[self.tris[:, 1]] - v0
        self.e2 = self.verts[self.tris[:, 2]] - v0

#
class Camera:
    def __init__(self, width=640, height=480, focal_px=400, pixel_noise_px=0.5, min_range_m=8.0):
        self.W, self.H, self.f = width, height, focal_px
        self.cx, self.cy = width / 2.0, height / 2.0
        self.pixel_noise_px = pixel_noise_px
        self.min_range_m = min_range_m      # reject near-collision poses

    def meta(self):
        return dict(width=self.W, height=self.H, focal_px=self.f,
                    pixel_noise_px=self.pixel_noise_px, min_range_m=self.min_range_m)


MODEL = AircraftModel()
CAM = Camera()
IMG_W, IMG_H, FOCAL = CAM.W, CAM.H, CAM.f
CX, CY = CAM.cx, CAM.cy
PIXEL_NOISE = CAM.pixel_noise_px
FEATURES_BODY = MODEL.keypoints
N_FEATURES = MODEL.N_FEATURES

# --------------------------------------------------------------------------- #
# Dynamics constants
# --------------------------------------------------------------------------- #
DT = 0.1            # s
LAT_GAIN = 10.0     # follower lateral velocity [m/s] = LAT_GAIN * sin(roll)   (simplified)

# --------------------------------------------------------------------------- #
# Pose-label discretisation (paper Table I): x, y, z, phi(roll)  -> 10*10*8*6 = 4800
# --------------------------------------------------------------------------- #
BIN_LO = np.array([-100.0, -50.0, -20.0, np.deg2rad(-45.0)])
BIN_HI = np.array([50.0, 50.0, 20.0, np.deg2rad(45.0)])
BIN_N = np.array([10, 10, 8, 6])
N_LABELS = int(np.prod(BIN_N))
POSE4_NAMES = ["x [m]", "y [m]", "z [m]", "phi [deg]"]

#
def pose_to_label(pose4):
    #Converts continuous (x,y,z,phi) values into one of 4,800 discrete labels.
    pose4 = np.atleast_2d(pose4)
    frac = (pose4[:, :4] - BIN_LO) / (BIN_HI - BIN_LO)
    idx = np.clip(np.floor(frac * BIN_N).astype(int), 0, BIN_N - 1)
    return np.ravel_multi_index(tuple(idx.T), tuple(BIN_N))

#
def label_to_index(labels):
    #Converts a label number into its four grid indices.
    return np.stack(np.unravel_index(np.asarray(labels), tuple(BIN_N)), axis=1)


def label_to_pose(labels):
    """Centre of the discretisation cell for each label -> (N, 4)."""
    return BIN_LO + (label_to_index(labels) + 0.5) / BIN_N * (BIN_HI - BIN_LO)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

#
def euler_to_R(phi, theta, psi):
    #Converts roll, pitch and yaw angles into rotation matrices.
    """ZYX Euler angles -> rotation matrices R = Rz(psi) Ry(theta) Rx(phi). (N,3,3)"""
    phi, theta, psi = (np.atleast_1d(np.asarray(a, float)) for a in (phi, theta, psi))
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    R = np.empty((phi.shape[0], 3, 3))
    R[:, 0, 0] = cth * cps
    R[:, 0, 1] = sph * sth * cps - cph * sps
    R[:, 0, 2] = cph * sth * cps + sph * sps
    R[:, 1, 0] = cth * sps
    R[:, 1, 1] = sph * sth * sps + cph * cps
    R[:, 1, 2] = cph * sth * sps - sph * cps
    R[:, 2, 0] = -sth
    R[:, 2, 1] = sph * cth
    R[:, 2, 2] = cph * cth
    return R


def _as6(states):
    s = np.atleast_2d(np.asarray(states, float))
    if s.shape[1] == 4:
        s = np.column_stack([s, np.zeros((len(s), 2))])
    return s

#
def camera_rotation(c_world):
    #Calculates the orientation of the camera relative to the aircraft coordinate frame.
    """Camera looks at the leader origin c_world (N,3). Columns of the result are the camera axes
    (right, down, forward) in the follower level frame; down is kept as close to world-down as possible."""
    fwd = c_world / np.linalg.norm(c_world, axis=1, keepdims=True)
    right = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    norm = np.linalg.norm(right, axis=1, keepdims=True)
    right = np.where(norm > 1e-9, right / np.maximum(norm, 1e-9), np.array([0.0, 1.0, 0.0]))
    cam_down = np.cross(fwd, right)
    return np.stack([right, cam_down, fwd], axis=2)

#
def _project_full(states, pts=None, cam=CAM):
    #Projects 3D aircraft keypoints into 2D pixels and also returns depth and geometric information.
    """Raw pinhole projection of body-frame points `pts` (K,3) for each state.
    Returns uv (N,K,2) [unclipped], z (N,K) camera depth, R_L (N,3,3), c_world (N,3)."""
    s = _as6(states)
    pts = MODEL.keypoints if pts is None else np.asarray(pts, float)
    c_world = -s[:, :3]
    R_L = euler_to_R(s[:, 3], s[:, 4], s[:, 5])
    kp_w = c_world[:, None, :] + np.einsum("nij,kj->nki", R_L, pts)
    kp_c = np.einsum("nji,nkj->nki", camera_rotation(c_world), kp_w)
    z = kp_c[..., 2]
    zs = np.maximum(z, 0.1)
    u = cam.cx + cam.f * kp_c[..., 0] / zs
    v = cam.cy + cam.f * kp_c[..., 1] / zs
    return np.stack([u, v], axis=-1), z, R_L, c_world


def project(states, pts=None, cam=CAM):
    """Unclipped keypoint pixels (N,K,2) - smooth in the state, used by particle filter and BA."""
    return _project_full(states, pts, cam)[0]

#
def _occluded(origin_b, kp_b, model, tol=0.05, chunk=1024):
    #Uses ray casting to check whether another part of the aircraft blocks a keypoint.
    """Ray cast from each origin (N,3) to every keypoint (N,K,3), both in the leader body frame.
    True where a mesh triangle blocks the line of sight (two-sided Moller-Trumbore)."""
    N, K, _ = kp_b.shape
    out = np.zeros((N, K), dtype=bool)
    v0, e1, e2 = model.v0, model.e1, model.e2
    for s in range(0, N, chunk):
        sl = slice(s, min(s + chunk, N))
        o = origin_b[sl][:, None, None, :]
        vec = kp_b[sl] - origin_b[sl][:, None, :]
        dist = np.linalg.norm(vec, axis=-1)
        d = (vec / dist[..., None])[:, :, None, :]
        pvec = np.cross(d, e2)
        det = np.sum(e1 * pvec, axis=-1)
        ok = np.abs(det) > 1e-9
        inv = 1.0 / np.where(ok, det, 1.0)
        tvec = o - v0
        u = np.sum(tvec * pvec, axis=-1) * inv
        qvec = np.cross(tvec, e1)
        v = np.sum(d * qvec, axis=-1) * inv
        t = np.sum(e2 * qvec, axis=-1) * inv
        eps = 1e-6
        hit = (ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > 1e-3)
               & (t < dist[..., None] - tol))
        out[sl] = hit.any(axis=-1)
    return out

#
def render_features(states, model=MODEL, cam=CAM, check_occlusion=True, rng=None, noise=False):
    #Generates the visible or occluded pixel features for given aircraft poses.
    """states (N,4|6) -> feats (N,14,3) = (u, v, occ); occ=1 means not detected (u = v = 0).
    check_occlusion=False skips the ray casting (cheap)."""
    s = _as6(states)
    uv, z, R_L, c_world = _project_full(s, model.keypoints, cam)
    u, v = uv[..., 0], uv[..., 1]
    if noise and cam.pixel_noise_px > 0:
        u = u + rng.normal(0, cam.pixel_noise_px, u.shape)
        v = v + rng.normal(0, cam.pixel_noise_px, v.shape)
    missing = (z <= 0.1) | (u < 0) | (u >= cam.W) | (v < 0) | (v >= cam.H)
    if check_occlusion:
        origin_b = np.einsum("nji,nj->ni", R_L, -c_world)      # camera position in leader body frame
        missing |= _occluded(origin_b, np.broadcast_to(model.keypoints, (len(s), model.N_FEATURES, 3)), model)
    occ = missing.astype(np.float32)
    return np.stack([np.where(missing, 0.0, u), np.where(missing, 0.0, v), occ], axis=-1).astype(np.float32)


def simulate_measurements(states, rng, pixel_noise=None, miss_prob=0.0):
    #Simulates the measurements that a feature detector would return, including missing points and pixel noise.
    """What an (assumed good) CNN feature detector returns: uv (N,14,2) with NaN where
    undetected, detected (N,14) bool."""
    cam = CAM if pixel_noise is None else Camera(CAM.W, CAM.H, CAM.f, pixel_noise, CAM.min_range_m)
    f = render_features(states, MODEL, cam, True, rng, noise=pixel_noise != 0)
    detected = f[..., 2] == 0
    if miss_prob > 0:
        detected &= rng.random(detected.shape) >= miss_prob
    uv = np.where(detected[..., None], f[..., :2], np.nan)
    return uv, detected

#
def encode_features(uv, detected):
    #Converts the 14 keypoints into the 42 input features used by the ANN and XGBoost models.
    """(N,14,2),(N,14) -> (N,42) network input: normalised [u, v, occluded_flag] per feature."""
    u = np.where(detected, (uv[..., 0] - CX) / CX, 0.0)
    v = np.where(detected, (uv[..., 1] - CY) / CY, 0.0)
    flag = (~detected).astype(float)
    return np.stack([u, v, flag], axis=-1).reshape(len(uv), -1).astype(np.float32)

#
def attitude_error_deg(est, true):
    #Calculates the angular difference between estimated and true aircraft orientations in degrees.
    """Geodesic angle [deg] between estimated and true leader attitude."""
    Re = euler_to_R(est[:, 3], est[:, 4], est[:, 5])
    Rt = euler_to_R(true[:, 3], true[:, 4], true[:, 5])
    tr = np.einsum("nij,nij->n", Re, Rt)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def position_error(est, true):
    #Calculates the 3D position error between the estimate and ground truth
    return np.linalg.norm(est[:, :3] - true[:, :3], axis=1)


# --------------------------------------------------------------------------- #
# Dynamics (simplified, discretised)
# --------------------------------------------------------------------------- #
def body_velocity(u):
    #Converts the follower's speed, pitch and roll controls into velocity components.
    """Follower controls u=[v_x, pitch, roll] -> velocity [vx, vy, vz] in the level frame.
    Pitch induces vertical velocity, roll induces lateral velocity (simplified)."""
    u = np.atleast_2d(u)
    v, th, ph = u[:, 0], u[:, 1], u[:, 2]
    return np.stack([v * np.cos(th), LAT_GAIN * np.sin(ph), -v * np.sin(th)], axis=1)

#
def propagate_position(states, uF, vL, dt=DT):
    #Predicts the next relative position using the follower and leader velocities.
    """p_{k+1} = p_k + dt (v_F - v_L).  v_F comes from the (known) follower controls; the leader
    flies along its own x axis at speed vL, so its velocity is R_L [vL, 0, 0]."""
    s = _as6(states)
    n = len(s)
    uF = np.broadcast_to(np.atleast_2d(uF), (n, 3))
    vL = np.broadcast_to(np.atleast_1d(np.asarray(vL, float)).reshape(-1), (n,))
    R_L = euler_to_R(s[:, 3], s[:, 4], s[:, 5])
    return s[:, :3] + dt * (body_velocity(uF) - vL[:, None] * R_L[:, :, 0])


# --------------------------------------------------------------------------- #
# Dataset + trajectory generation
# --------------------------------------------------------------------------- #
def get_dirs(out_dir="outputs"):
    root = Path(out_dir)
    d = {k: root / k for k in ("data", "models", "figures", "results")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def sample_states(n, rng, min_range=None, att_jitter_deg=0.0):
    #Randomly generates valid aircraft poses, rejecting poses that violate the minimum distance constraint.
    """Uniform samples in the Table I box; poses closer than min_range are rejected."""
    min_range = CAM.min_range_m if min_range is None else min_range
    out = np.empty((0, 6))
    jit = np.deg2rad(att_jitter_deg)
    while len(out) < n:
        m = int((n - len(out)) * 1.2) + 16
        s = np.column_stack([rng.uniform(BIN_LO, BIN_HI, size=(m, 4)),
                             np.clip(rng.normal(0, jit, (m, 2)), -np.deg2rad(10), np.deg2rad(10))])
        out = np.concatenate([out, s[np.linalg.norm(s[:, :3], axis=1) >= min_range]])
    return out[:n]


def generate_dataset(n_samples, seed, pixel_noise=None, att_jitter_deg=0.0, min_range=8,
                     val_frac=0.15, test_frac=0.15, block=20000, verbose=True):
    #Creates synthetic samples, simulates their image features, assigns labels and splits the data into training, validation and test sets.
    rng = np.random.default_rng(seed)
    states = sample_states(n_samples, rng, min_range, att_jitter_deg)
    uvs, dets = [], []
    for i in range(0, n_samples, block):
        uv, det = simulate_measurements(states[i:i + block], rng, pixel_noise)
        uvs.append(uv)
        dets.append(det)
        if verbose:
            print(f"  rendered {min(i + block, n_samples)}/{n_samples}", end="\r", flush=True)
    if verbose:
        print()
    uv, det = np.concatenate(uvs), np.concatenate(dets)
    labels = pose_to_label(states[:, :4])
    perm = rng.permutation(n_samples)
    n_va, n_te = int(val_frac * n_samples), int(test_frac * n_samples)
    split = np.zeros(n_samples, np.int8)      # 0 train, 1 val, 2 test
    split[perm[:n_va]] = 1
    split[perm[n_va:n_va + n_te]] = 2
    meta = dict(cam=CAM.meta(), seed=seed, att_jitter_deg=att_jitter_deg,
                grid_min=BIN_LO.tolist(), grid_max=BIN_HI.tolist(), grid_n=BIN_N.tolist())
    return dict(states=states, uv=uv.astype(np.float32), detected=det, labels=labels, split=split,
                features_body=MODEL.keypoints, camera=np.array([CAM.W, CAM.f, CAM.H]),
                meta_json=np.array(json.dumps(meta)))


def generate_trajectory(seed, duration=10.0, dt=DT, p0=(-90.0, 25.0, 12.0), pixel_noise=None):
    #Generates a time-based formation-flight scenario, including ground-truth poses and camera measurements.
    """Formation-approach scenario: follower closes ~5 m/s from 90 m behind the leader
    (cf. paper Fig. 6).  Leader attitude is scripted; positions are integrated with the dynamics."""
    rng = np.random.default_rng(seed + 1)
    K = int(round(duration / dt)) + 1
    t = np.arange(K) * dt
    att = np.stack([np.deg2rad(10) * np.sin(2 * np.pi * t / 6.0),
                    np.deg2rad(2) * np.sin(2 * np.pi * t / 5.0 + 1.0),
                    np.deg2rad(1.5) * np.sin(2 * np.pi * t / 8.0)], axis=1)
    uF = np.tile([30.0, 0.0, np.arcsin(-1.0 / LAT_GAIN)], (K, 1))
    vL = np.full(K, 25.0)
    states = np.zeros((K, 6))
    states[:, 3:] = att
    states[0, :3] = p0
    for k in range(K - 1):
        states[k + 1, :3] = propagate_position(states[k:k + 1], uF[k], vL[k], dt)[0]
    uv, det = simulate_measurements(states, rng, pixel_noise)
    return dict(times=t, states=states, uF=uF, vL=vL, uv=uv, detected=det)


def load_dataset(out_dir="outputs"):
    #Loads the saved training dataset.
    d = np.load(get_dirs(out_dir)["data"] / "dataset.npz")
    return {k: d[k] for k in d.files}


def load_trajectory(out_dir="outputs"):
    #Loads the saved flight trajectory.
    d = np.load(get_dirs(out_dir)["data"] / "trajectory.npz")
    return {k: d[k] for k in d.files}


def save_result(path, res):
    #Saves result arrays in an NPZ file.
    np.savez(path, **{k: np.asarray(v) for k, v in res.items()})


def load_result(path):
    #Loads results from an NPZ file.
    d = np.load(path)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# --------------------------------------------------------------------------- #
# Rendering + figures
# --------------------------------------------------------------------------- #
def fig_trajectory(traj, path):
    #Plots the aircraft's relative position over time.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for j, nm in enumerate("xyz"):
        ax.plot(traj["times"], traj["states"][:, j], label=nm)
    ax.set_xlabel("time")
    ax.set_ylabel("relative position [m]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_error_traces(times, pos_err, att_err, path, title=None, baseline=None,
                      baseline_label="coarse estimate only"):
    """Paper Fig. 9/10 style: position error [m] and attitude error [deg] vs time."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    for ax, e, ylab, bi in ((axes[0], pos_err, "position error [m]", 0),
                            (axes[1], att_err, "attitude error [deg]", 1)):
        ax.plot(times, e, color="#1f9bf0", lw=1.2, label="estimate")
        if baseline is not None:
            ax.plot(times, baseline[bi], "--", color="gray", lw=0.9, alpha=0.8, label=baseline_label)
        ax.set_ylabel(ylab)
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    if baseline is not None:
        axes[0].legend(fontsize=8)
    if title:
        fig.text(0.5, 0.005, title, ha="center", va="bottom")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_hists(err, path, bins=60):
    """2x2 histogram grid for (x, y, z, phi) errors (paper Fig. 4/5 layout)."""
    err = err.copy()
    err[:, 3] = np.degrees(err[:, 3])
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    for j, ax in enumerate(axes.ravel()):
        ax.hist(err[:, j], bins=bins, color="#1f9bf0", edgecolor="k", linewidth=0.3)
        ax.set_xlabel(POSE4_NAMES[j])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(dirs, traj):
    fd = dirs["figures"]
    fig_trajectory(traj, fd / "fig06_relative_position_trajectory.png")


def generate_all(out_dir="outputs", n_samples=1_000_000, seed=0, figures=True,
                 att_jitter_deg=0.0, min_range=None):
    dirs = get_dirs(out_dir)
    print(f"[dataset] generating {n_samples} samples (seed {seed}) ...")
    ds = generate_dataset(n_samples, seed, att_jitter_deg=att_jitter_deg, min_range=min_range)
    np.savez_compressed(dirs["data"] / "dataset.npz", **ds)
    traj = generate_trajectory(seed)
    np.savez_compressed(dirs["data"] / "trajectory.npz", **traj)
    occ = 1.0 - ds["detected"].mean()
    stats = dict(n_samples=n_samples, n_labels_used=int(len(np.unique(ds["labels"]))),
                 n_labels_total=N_LABELS, mean_features_detected=float(ds["detected"].sum(1).mean()),
                 mean_fraction_not_detected=float(occ),
                 frac_samples_no_feature_detected=float((ds["detected"].sum(1) == 0).mean()),
                 camera=CAM.meta(), att_jitter_deg=att_jitter_deg)
    (dirs["results"] / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print("[dataset]", stats)
    if figures:
        make_figures(dirs, traj)
        print(f"[dataset] figure 6 written to {dirs['figures']}")
    return ds, traj


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n-samples", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--att-jitter-deg", type=float, default=0.0,
                    help="std of random leader pitch/yaw [deg]; 0 = ignored, as in the paper's labels")
    ap.add_argument("--min-range", type=float, default=8, help="closest sampled distance [m] (default 8)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    generate_all(a.out, a.n_samples, a.seed, not a.no_figures, a.att_jitter_deg, a.min_range)
