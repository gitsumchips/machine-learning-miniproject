"""
train_ann.py  --  Pipeline 1, step 1: coarse pose classifier (paper Sec. IV-D)
==============================================================================
MLP: 42 inputs (u, v, occlusion flag for each of the 14 features) -> 100 -> 100 (ReLU)
     -> 4800-way softmax (cross-entropy loss), trained with Adam.

Outputs (in <out>/):
  models/ann.pt                    trained weights
  data/ann_error_dist.npz          (true - classified) pose errors on the validation set;
                                   the particle filter samples from this distribution
  results/ann_metrics.json         validation / test accuracy etc. (paper reports 54 % val acc.)
  figures/fig04_ann_error_vs_true_pose.png      paper Fig. 4
  figures/fig05_ann_error_vs_correct_label.png  paper Fig. 5
  figures/ann_training_curve.png   (extra)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset_generation as dg


class PoseMLP(nn.Module):
    def __init__(self, n_in=3 * dg.N_FEATURES, hidden=100, n_out=dg.N_LABELS):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))   # softmax is inside the CE loss

    def forward(self, x):
        return self.net(x)


def load_ann(out_dir="outputs", device="cpu"):
    ck = torch.load(dg.get_dirs(out_dir)["models"] / "ann.pt", map_location=device)
    m = PoseMLP(ck["n_in"], ck["hidden"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def predict_labels(model, X, batch=16384):
    dev = next(model.parameters()).device
    model.eval()
    out = []
    Xt = torch.as_tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch):
        out.append(model(Xt[i:i + batch].to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


def run(out_dir="outputs", epochs=40, batch=256, lr=1e-3, seed=0, device=None):
    dirs = dg.get_dirs(out_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = dg.load_dataset(out_dir)
    X = dg.encode_features(data["uv"], data["detected"])
    y, split, states = data["labels"], data["split"], data["states"]
    tr, va, te = (np.where(split == s)[0] for s in (0, 1, 2))
    Xt = {k: torch.as_tensor(X[i]).to(device) for k, i in (("tr", tr), ("va", va), ("te", te))}
    yt = {k: torch.as_tensor(y[i]).long().to(device) for k, i in (("tr", tr), ("va", va), ("te", te))}

    model = PoseMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    @torch.no_grad()
    def acc(k):
        model.eval()
        return float((model(Xt[k]).argmax(1) == yt[k]).float().mean())

    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt["tr"]), device=device)
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt["tr"][idx]), yt["tr"][idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        va_acc = acc("va")
        hist.append((tot / len(perm), va_acc))
        if va_acc > best:
            best, best_state = va_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[ann] epoch {ep + 1:3d}/{epochs}  loss {tot / len(perm):.4f}  val acc {va_acc:.4f}")
    model.load_state_dict(best_state)

    torch.save(dict(state_dict=best_state, n_in=X.shape[1], hidden=100), dirs["models"] / "ann.pt")

    # ---- evaluation on validation set (as in the paper) ----
    pred_va = predict_labels(model, X[va])
    cls_pose = dg.label_to_pose(pred_va)
    true_pose = states[va][:, :4]
    label_pose = dg.label_to_pose(y[va])
    err_true = cls_pose - true_pose            # Fig. 4
    err_label = cls_pose - label_pose          # Fig. 5
    dg.plot_error_hists(err_true, dirs["figures"] / "fig04_ann_error_vs_true_pose.png")
    dg.plot_error_hists(err_label, dirs["figures"] / "fig05_ann_error_vs_correct_label.png")

    # error distribution used by the particle filter: true = classified + delta
    np.savez(dirs["data"] / "ann_error_dist.npz", delta=(true_pose - cls_pose))

    adj = np.all(np.abs(dg.label_to_index(pred_va) - dg.label_to_index(y[va])) <= 1, axis=1).mean()
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot([h[0] for h in hist]); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss")
    ax[1].plot([h[1] for h in hist]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy")
    fig.tight_layout(); fig.savefig(dirs["figures"] / "ann_training_curve.png", dpi=150); plt.close(fig)

    rmse = np.sqrt(np.mean(err_true ** 2, axis=0))
    metrics = dict(val_accuracy=float(best), test_accuracy=acc("te"),
                   val_adjacent_cell_accuracy=float(adj), n_train=len(tr), n_val=len(va),
                   n_test=len(te), epochs=epochs,
                   val_pose_rmse=dict(x_m=rmse[0], y_m=rmse[1], z_m=rmse[2], phi_deg=np.degrees(rmse[3])),
                   val_pos_rmse_3d_m=float(np.sqrt(np.mean(np.sum(err_true[:, :3] ** 2, axis=1)))))
    (dirs["results"] / "ann_metrics.json").write_text(json.dumps(dg.to_jsonable(metrics), indent=2))
    print("[ann]", json.dumps(dg.to_jsonable(metrics), indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.out, a.epochs, a.batch, a.lr, a.seed)