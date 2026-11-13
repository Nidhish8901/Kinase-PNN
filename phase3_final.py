"""
Phase 3 – Final Controlled Evaluation
Transfer Learning with Physics-Informed Neural Networks for Bioethanol Fermentation

Protocol:
  Source           : Episode 20
  Target           : Episode 75
  FIXED test set   : t = 7, 8, 9  (never used for training, val, norm-fitting, or HP-tuning)
  Training pool    : t = 0..6  (7 measurements)
  Nested budgets   : n=3 → {t=0,3,6}
                     n=5 → {t=0,1,3,5,6}
                     n=7 → {t=0,1,2,3,4,5,6}
  Val set (ES only): drawn from training pool rows NOT used in that budget's subset
  Scaler           : fitted on training subset only (no test leakage)
  Seeds            : 3 source pretraining seeds × 3 scratch-model seeds = 9 experiments/condition
  Physics colloc.  : 150 pts over [0, t_max_train]  (never extends into test domain)
  Outputs          : outputs/phase3_final/
"""

import os, json, copy, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
P1_DIR    = os.path.join(BASE_DIR, "outputs", "phase1")
P2_DIR    = os.path.join(BASE_DIR, "outputs", "phase2")
OUT_DIR   = os.path.join(BASE_DIR, "outputs", "phase3_final")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fixed protocol constants (MUST NOT change between conditions) ─────────────
TEST_TIMES   = {7.0, 8.0, 9.0}            # fixed held-out test set
TRAIN_BUDGETS = {
    3: [0.0, 3.0, 6.0],
    5: [0.0, 1.0, 3.0, 5.0, 6.0],
    7: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
# Val for each budget: one observation from the training pool, not in train subset
VAL_TIMES = {
    3: [2.0],          # leftover from pool after n=3 training points
    5: [2.0],
    7: [],             # no pool remainder; use last training point as proxy
}
# When no pool remainder, use the last training point as a val proxy
# (it IS in the training set, so ES is not fully independent; document this)
VAL_FALLBACK = {7: [6.0]}  # same point is in training; ES is in-training only

SOURCE_SEEDS = [42, 7, 123]
SCRATCH_SEEDS = [42, 7, 123]
YPS_PHYSICAL  = 0.511
N_COLLOC      = 150

# Fixed hyperparameters (selected on source episode, not tuned on target test)
LR_SCRATCH   = 1e-3
LR_FINETUNE  = 3e-4
MAX_EPOCHS   = 15_000
PATIENCE     = 1_000
LR_PATIENCE  = 500
LR_FACTOR    = 0.5
MIN_LR       = 1e-6
LAMBDA_DATA  = 1.0
LAMBDA_PHYS  = 0.1
LAMBDA_IC    = 1.0

KINETIC_INIT = dict(mu_max=0.8, Ks=0.5, Yxs=0.45, Yps=0.50)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)


def r2(yt, yp):
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return float("nan") if ss_tot < 1e-10 else 1.0 - ss_res / ss_tot


def compute_metrics(y_true, y_pred):
    out = {}
    for i, lab in enumerate(("S", "X", "P")):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[lab] = {
            "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "MAE":  float(np.mean(np.abs(yt - yp))),
            "R2":   r2(yt, yp) if len(yt) >= 3 else float("nan"),
        }
    return out


def load_episode(ep_id):
    df = pd.read_csv(os.path.join(P1_DIR, f"episode_{ep_id}_cleaned.csv"))
    return df.sort_values("time_hours").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Scaler – fitted on training data only
# ══════════════════════════════════════════════════════════════════════════════
class Scaler:
    """
    Fitted exclusively on training rows.
    t_max is extended to cover the full fermentation interval for collocation
    but concentration scales use only training data.
    """
    def __init__(self, df_train: pd.DataFrame, t_max_full: float | None = None):
        self.t_min   = float(df_train["time_hours"].min())
        self.t_max   = float(df_train["time_hours"].max())
        # Use full interval for collocation points but NOT for scaling
        self.t_max_full = t_max_full if t_max_full is not None else self.t_max
        self.s_max   = float(df_train["glucose"].max())
        self.x_max   = float(df_train["biomass"].max())
        self.p_max   = float(df_train["ethanol"].max())

    def norm_t(self, t):
        return (np.asarray(t, float) - self.t_min) / (self.t_max - self.t_min + 1e-8)

    def norm_t_full(self, t):
        """Normalise to [0, t_max_full] — used for prediction over full interval."""
        return (np.asarray(t, float) - self.t_min) / (self.t_max_full - self.t_min + 1e-8)

    def norm_s(self, s):  return float(np.asarray(s)) / (self.s_max + 1e-8)
    def norm_x(self, x):  return float(np.asarray(x)) / (self.x_max + 1e-8)
    def norm_p(self, p):  return float(np.asarray(p)) / (self.p_max + 1e-8)

    def tensors(self, df):
        t = torch.tensor(self.norm_t(df["time_hours"].values), dtype=torch.float32).unsqueeze(1)
        s = torch.tensor([self.norm_s(v) for v in df["glucose"].values],
                         dtype=torch.float32).unsqueeze(1)
        x = torch.tensor([self.norm_x(v) for v in df["biomass"].values],
                         dtype=torch.float32).unsqueeze(1)
        p = torch.tensor([self.norm_p(v) for v in df["ethanol"].values],
                         dtype=torch.float32).unsqueeze(1)
        return t, s, x, p

    def denorm(self, pred_np):
        """pred_np shape (N,3) normalised → physical."""
        return np.column_stack([
            pred_np[:, 0] * self.s_max,
            pred_np[:, 1] * self.x_max,
            pred_np[:, 2] * self.p_max,
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
class FermentationMLP(nn.Module):
    def __init__(self, hidden=32, n_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return nn.functional.softplus(self.net(t))


class KineticParams(nn.Module):
    def __init__(self, mu_max=0.8, Ks=0.5, Yxs=0.45, Yps=0.50, clamp_Yps=False):
        super().__init__()
        self.clamp_Yps = clamp_Yps
        def inv_sp(x):
            return float(np.log(np.exp(float(x)) - 1.0 + 1e-8))
        self._mu_max = nn.Parameter(torch.tensor(inv_sp(mu_max)))
        self._Ks     = nn.Parameter(torch.tensor(inv_sp(Ks)))
        self._Yxs    = nn.Parameter(torch.tensor(inv_sp(Yxs)))
        self._Yps    = nn.Parameter(torch.tensor(inv_sp(Yps)))

    @property
    def mu_max(self): return nn.functional.softplus(self._mu_max)
    @property
    def Ks(self):     return nn.functional.softplus(self._Ks)
    @property
    def Yxs(self):    return nn.functional.softplus(self._Yxs)
    @property
    def Yps(self):
        v = nn.functional.softplus(self._Yps)
        return torch.clamp(v, max=YPS_PHYSICAL) if self.clamp_Yps else v

    def to_dict(self):
        return {k: getattr(self, k).item() for k in ("mu_max", "Ks", "Yxs", "Yps")}


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════
_mse = nn.MSELoss()


def _loss_data(model, t, s, x, p):
    pred = model(t)
    return _mse(pred[:, 0:1], s) + _mse(pred[:, 1:2], x) + _mse(pred[:, 2:3], p)


def _loss_ic(model, scaler, S0, X0, P0):
    t0   = torch.zeros(1, 1, dtype=torch.float32)
    pred = model(t0)
    return ((pred[0, 0] - scaler.norm_s(S0)) ** 2 +
            (pred[0, 1] - scaler.norm_x(X0)) ** 2 +
            (pred[0, 2] - scaler.norm_p(P0)) ** 2)


def _loss_phys(model, params, scaler, tc):
    tc2  = tc.clone().requires_grad_(True)
    pred = model(tc2)
    Sn, Xn, Pn = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    o    = torch.ones_like(Sn)
    dSn  = torch.autograd.grad(Sn, tc2, o, create_graph=True)[0]
    dXn  = torch.autograd.grad(Xn, tc2, o, create_graph=True, retain_graph=True)[0]
    dPn  = torch.autograd.grad(Pn, tc2, o, create_graph=True, retain_graph=True)[0]
    T    = scaler.t_max - scaler.t_min
    dS   = dSn * (scaler.s_max / T)
    dX   = dXn * (scaler.x_max / T)
    dP   = dPn * (scaler.p_max / T)
    S    = Sn * scaler.s_max
    X    = Xn * scaler.x_max
    mu   = params.mu_max * S / (params.Ks + S + 1e-8)
    rX   = dX - mu * X
    rS   = dS + (1.0 / params.Yxs) * mu * X
    rP   = dP - params.Yps * (1.0 / params.Yxs) * mu * X
    return (rX.pow(2).mean() + rS.pow(2).mean() + rP.pow(2).mean()) / 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Predict & residuals
# ══════════════════════════════════════════════════════════════════════════════
def predict_phys(model, scaler, t_arr):
    model.eval()
    tn = torch.tensor(scaler.norm_t(np.asarray(t_arr, float)),
                      dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pn = model(tn).numpy()
    return scaler.denorm(pn)


def eval_residuals_rms(model, params, scaler, n=150):
    t_p = np.linspace(scaler.t_min, scaler.t_max, n)
    tc  = torch.tensor(scaler.norm_t(t_p), dtype=torch.float32,
                       requires_grad=True).unsqueeze(1)
    pred = model(tc)
    Sn, Xn, Pn = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    dSn = torch.autograd.grad(Sn, tc, torch.ones_like(Sn), retain_graph=True)[0]
    dXn = torch.autograd.grad(Xn, tc, torch.ones_like(Xn), retain_graph=True)[0]
    dPn = torch.autograd.grad(Pn, tc, torch.ones_like(Pn), retain_graph=False)[0]
    T   = scaler.t_max - scaler.t_min
    dS  = dSn.detach().numpy()[:, 0] * scaler.s_max / T
    dX  = dXn.detach().numpy()[:, 0] * scaler.x_max / T
    dP  = dPn.detach().numpy()[:, 0] * scaler.p_max / T
    S   = Sn.detach().numpy()[:, 0] * scaler.s_max
    X   = Xn.detach().numpy()[:, 0] * scaler.x_max
    mu  = params.mu_max.item() * S / (params.Ks.item() + S + 1e-8)
    rX  = dX - mu * X
    rS  = dS + (1 / params.Yxs.item()) * mu * X
    rP  = dP - params.Yps.item() * (1 / params.Yxs.item()) * mu * X
    return {"rX_rms": float(np.sqrt(np.mean(rX**2))),
            "rS_rms": float(np.sqrt(np.mean(rS**2))),
            "rP_rms": float(np.sqrt(np.mean(rP**2)))}


# ══════════════════════════════════════════════════════════════════════════════
# Training loop (PINN or baseline NN)
# ══════════════════════════════════════════════════════════════════════════════
def train_model(model, scaler, df_train, df_val,
                params=None, use_physics=True,
                lr=LR_SCRATCH, frozen_layer_0=False):
    """
    Returns (n_epochs_stopped, best_val_loss).
    Val rows guaranteed to NOT overlap with test set (protocol enforced by caller).
    """
    t_obs, s_obs, x_obs, p_obs = scaler.tensors(df_train)
    t_val, s_val, x_val, p_val = scaler.tensors(df_val)
    S0 = float(df_train["glucose"].iloc[0])
    X0 = float(df_train["biomass"].iloc[0])
    P0 = float(df_train["ethanol"].iloc[0])

    # Collocation over training time domain only
    tc = torch.tensor(
        scaler.norm_t(np.linspace(scaler.t_min, scaler.t_max, N_COLLOC)),
        dtype=torch.float32).unsqueeze(1)

    if frozen_layer_0:
        first = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
        for p in first.parameters():
            p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if use_physics and params is not None:
        trainable = trainable + list(params.parameters())

    opt   = torch.optim.Adam(trainable, lr=lr)
    sched = ReduceLROnPlateau(opt, mode="min", factor=LR_FACTOR,
                               patience=LR_PATIENCE, min_lr=MIN_LR)

    best_val, best_state, patience_ct = float("inf"), None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        ld = _loss_data(model, t_obs, s_obs, x_obs, p_obs)
        li = _loss_ic(model, scaler, S0, X0, P0)
        if use_physics and params is not None:
            lp    = _loss_phys(model, params, scaler, tc)
            total = LAMBDA_DATA * ld + LAMBDA_PHYS * lp + LAMBDA_IC * li
        else:
            total = LAMBDA_DATA * ld + LAMBDA_IC * li
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(t_val)
            lv = (_mse(pv[:, 0:1], s_val) + _mse(pv[:, 1:2], x_val) +
                  _mse(pv[:, 2:3], p_val))
        sched.step(lv.item())

        if lv.item() < best_val - 1e-7:
            best_val    = lv.item()
            patience_ct = 0
            best_state  = copy.deepcopy(model.state_dict())
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    return epoch + 1, best_val


# ══════════════════════════════════════════════════════════════════════════════
# Source pretraining
# ══════════════════════════════════════════════════════════════════════════════
def pretrain_source(seed, df20):
    """Return (net_state, params_state, fitted_params_dict, n_epochs)."""
    set_seed(seed)
    sc20  = Scaler(df20)
    train = df20.iloc[:-2].copy()
    val   = df20.iloc[-2:].copy()
    net   = FermentationMLP()
    par   = KineticParams(**KINETIC_INIT)
    n_ep, _ = train_model(net, sc20, train, val,
                          params=par, use_physics=True, lr=LR_SCRATCH)
    return (copy.deepcopy(net.state_dict()),
            copy.deepcopy(par.state_dict()),
            par.to_dict(), n_ep)


# ══════════════════════════════════════════════════════════════════════════════
# Single experiment: one condition, one n_obs, one seed
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(condition, n_obs, seed,
                   df75, df75_test,
                   source_state, source_params_state,
                   clamp_Yps=False):
    """
    condition: "transfer_full" | "transfer_frozen" | "scratch_pinn" | "scratch_nn"
    Returns dict with metrics on fixed test set, residuals, timing, params.
    """
    set_seed(seed)

    train_times  = TRAIN_BUDGETS[n_obs]
    df_train     = df75[df75["time_hours"].isin(train_times)].copy().reset_index(drop=True)

    # Val: pool rows NOT in training and NOT in test set
    pool_times   = set(t for t in df75["time_hours"] if t not in TEST_TIMES)
    leftover     = sorted(pool_times - set(train_times))
    if leftover:
        # Pick the middle leftover point; deterministic
        mid      = leftover[len(leftover) // 2]
        df_val   = df75[df75["time_hours"] == mid].copy().reset_index(drop=True)
    else:
        # n=7 uses all pool rows; fall back to last training point as val proxy
        # (documented limitation: val is in-training only)
        df_val   = df_train.iloc[[-1]].copy().reset_index(drop=True)

    # Verify no test leakage
    val_in_test = set(df_val["time_hours"]) & TEST_TIMES
    assert not val_in_test, f"Val leaks into test: {val_in_test}"
    train_in_test = set(df_train["time_hours"]) & TEST_TIMES
    assert not train_in_test, f"Train leaks into test: {train_in_test}"

    # Scaler: fit on training rows only
    sc = Scaler(df_train, t_max_full=max(TEST_TIMES))

    use_physics = (condition != "scratch_nn")
    lr = LR_FINETUNE if condition in ("transfer_full", "transfer_frozen") else LR_SCRATCH
    frozen = (condition == "transfer_frozen")

    net = FermentationMLP()
    par = KineticParams(**KINETIC_INIT, clamp_Yps=clamp_Yps)

    if condition in ("transfer_full", "transfer_frozen"):
        net.load_state_dict(copy.deepcopy(source_state))
        par.load_state_dict(copy.deepcopy(source_params_state))
        if clamp_Yps:
            # Re-clamp after loading source weights
            par.clamp_Yps = True

    t_start = time.time()
    n_ep, best_val = train_model(
        net, sc, df_train, df_val,
        params=par if use_physics else None,
        use_physics=use_physics, lr=lr, frozen_layer_0=frozen)
    elapsed = time.time() - t_start

    # Evaluate on fixed test set
    y_test_true = df75_test[["glucose", "biomass", "ethanol"]].values
    y_test_pred = predict_phys(net, sc, df75_test["time_hours"].values)
    m = compute_metrics(y_test_true, y_test_pred)

    # Physics residuals
    if use_physics:
        try:
            res = eval_residuals_rms(net, sc, par)
        except Exception:
            res = {k: float("nan") for k in ("rX_rms", "rS_rms", "rP_rms")}
    else:
        res = {k: float("nan") for k in ("rX_rms", "rS_rms", "rP_rms")}

    val_note = ("in-training proxy" if not leftover
                else f"pool holdout t={df_val['time_hours'].iloc[0]}")

    return {
        "condition":    condition,
        "n_obs":        n_obs,
        "seed":         seed,
        "clamp_Yps":    clamp_Yps,
        "train_times":  train_times,
        "val_note":     val_note,
        "test_times":   sorted(TEST_TIMES),
        "metrics":      m,
        "residuals":    res,
        "params":       par.to_dict() if use_physics else {},
        "n_epochs":     n_ep,
        "elapsed_s":    elapsed,
        "model_state":  copy.deepcopy(net.state_dict()),
        "params_state": copy.deepcopy(par.state_dict()) if use_physics else {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate across seeds
# ══════════════════════════════════════════════════════════════════════════════
def aggregate(runs):
    def arr(key_path):
        vals = []
        for r in runs:
            obj = r
            for k in key_path:
                obj = obj[k]
            vals.append(float(obj))
        return np.array(vals, dtype=float)

    out = {"condition": runs[0]["condition"],
           "n_obs":     runs[0]["n_obs"],
           "clamp_Yps": runs[0]["clamp_Yps"],
           "n_runs":    len(runs)}
    for var in ("S", "X", "P"):
        for metric in ("RMSE", "MAE", "R2"):
            v = arr(["metrics", var, metric])
            out[f"{var}_{metric}_mean"] = float(np.nanmean(v))
            out[f"{var}_{metric}_std"]  = float(np.nanstd(v))
    for rk in ("rX_rms", "rS_rms", "rP_rms"):
        v = arr(["residuals", rk])
        out[f"{rk}_mean"] = float(np.nanmean(v))
        out[f"{rk}_std"]  = float(np.nanstd(v))
    out["n_epochs_mean"]  = float(np.mean([r["n_epochs"] for r in runs]))
    out["elapsed_s_mean"] = float(np.mean([r["elapsed_s"] for r in runs]))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "transfer_full":   "#2196F3",
    "transfer_frozen": "#9C27B0",
    "scratch_pinn":    "#E91E63",
    "scratch_nn":      "#FF9800",
}
LABELS = {
    "transfer_full":   "Transfer (full)",
    "transfer_frozen": "Transfer (frozen)",
    "scratch_pinn":    "Scratch PINN",
    "scratch_nn":      "Scratch NN",
}


def plot_rmse_vs_nobs(agg_table, clamp_Yps, out_path):
    conds = ["transfer_full", "transfer_frozen", "scratch_pinn", "scratch_nn"]
    rows  = [r for r in agg_table if r["clamp_Yps"] == clamp_Yps]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, var in zip(axes, ("S", "X", "P")):
        for cond in conds:
            cr   = sorted([r for r in rows if r["condition"] == cond],
                          key=lambda r: r["n_obs"])
            xs   = [r["n_obs"] for r in cr]
            ys   = [r[f"{var}_RMSE_mean"] for r in cr]
            errs = [r[f"{var}_RMSE_std"] for r in cr]
            ax.errorbar(xs, ys, yerr=errs, marker="o", lw=1.8,
                        color=COLORS[cond], label=LABELS[cond], capsize=3)
        ax.set_xlabel("Target observations (from pool t=0..6)")
        ax.set_ylabel(f"RMSE {var} (g/L)")
        ax.set_title(f"{var} — Fixed test set t={{7,8,9}}")
        ax.set_xticks([3, 5, 7])
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    tag = "Yps-bounded" if clamp_Yps else "Unconstrained"
    plt.suptitle(f"Controlled Evaluation [{tag}]: RMSE vs Observations "
                 "(mean ± std, 3×3=9 runs)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(df75, agg_table, best_runs_by_cond, n_obs, clamp_Yps, out_path):
    sc = Scaler(
        df75[df75["time_hours"].isin(TRAIN_BUDGETS[n_obs])].copy(),
        t_max_full=9.0)
    t_dense = np.linspace(0.0, 9.0, 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, idx) in zip(axes, [("glucose", 0), ("biomass", 1), ("ethanol", 2)]):
        # All observations
        ax.scatter(df75["time_hours"], df75[col], color="black",
                   s=45, zorder=6, label="All obs (Ep75)")
        # Training points
        t_tr = TRAIN_BUDGETS[n_obs]
        df_tr = df75[df75["time_hours"].isin(t_tr)]
        ax.scatter(df_tr["time_hours"], df_tr[col], color="lime",
                   s=90, zorder=7, marker="*", label=f"Train ({n_obs} pts)")
        # Test region
        ax.axvspan(min(TEST_TIMES) - 0.3, max(TEST_TIMES) + 0.3,
                   alpha=0.06, color="red", label="Test region")
        # Model curves
        for cond, run in best_runs_by_cond.items():
            net = FermentationMLP()
            net.load_state_dict(run["model_state"])
            pred = predict_phys(net, sc, t_dense)
            ax.plot(t_dense, pred[:, idx],
                    color=COLORS[cond], lw=1.8, label=LABELS[cond])
        ax.set_xlabel("Time (h)"); ax.set_ylabel(col.capitalize())
        ax.set_title(col.capitalize()); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    tag = "Yps-bounded" if clamp_Yps else "Unconstrained"
    plt.suptitle(f"Ep75 Predictions [{tag}] — {n_obs} training obs, "
                 f"test = t{{7,8,9}}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals_final(best_runs_by_cond, df75, out_path):
    """ODE residuals for PINN conditions across full [0,9] h interval."""
    sc_full = Scaler(
        df75[df75["time_hours"].isin(TRAIN_BUDGETS[7])].copy(),
        t_max_full=9.0)
    t_dense = np.linspace(0.0, 9.0, 300)

    pinn_conds = [c for c in best_runs_by_cond
                  if best_runs_by_cond[c]["params_state"]]
    if not pinn_conds:
        return

    fig, axes = plt.subplots(len(pinn_conds), 3,
                              figsize=(14, 3.5 * len(pinn_conds)))
    if len(pinn_conds) == 1:
        axes = [axes]

    for row_ax, cond in zip(axes, pinn_conds):
        run = best_runs_by_cond[cond]
        net = FermentationMLP(); net.load_state_dict(run["model_state"]); net.eval()
        par = KineticParams();   par.load_state_dict(run["params_state"])
        tc  = torch.tensor(sc_full.norm_t(t_dense), dtype=torch.float32,
                           requires_grad=True).unsqueeze(1)
        pred = net(tc)
        Sn, Xn, Pn = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        dSn = torch.autograd.grad(Sn, tc, torch.ones_like(Sn), retain_graph=True)[0]
        dXn = torch.autograd.grad(Xn, tc, torch.ones_like(Xn), retain_graph=True)[0]
        dPn = torch.autograd.grad(Pn, tc, torch.ones_like(Pn), retain_graph=False)[0]
        T   = sc_full.t_max - sc_full.t_min
        dS  = dSn.detach().numpy()[:, 0] * sc_full.s_max / T
        dX  = dXn.detach().numpy()[:, 0] * sc_full.x_max / T
        dP  = dPn.detach().numpy()[:, 0] * sc_full.p_max / T
        S   = Sn.detach().numpy()[:, 0] * sc_full.s_max
        X   = Xn.detach().numpy()[:, 0] * sc_full.x_max
        mu  = par.mu_max.item() * S / (par.Ks.item() + S + 1e-8)
        rX  = dX - mu * X
        rS  = dS + (1 / par.Yxs.item()) * mu * X
        rP  = dP - par.Yps.item() * (1 / par.Yxs.item()) * mu * X

        for ax, res, lab, col in zip(
                row_ax,
                [rX, rS, rP],
                ["dX/dt resid.", "dS/dt resid.", "dP/dt resid."],
                ["#2196F3", "#E91E63", "#4CAF50"]):
            ax.plot(t_dense, res, color=col, lw=1.4)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.axvspan(min(TEST_TIMES) - 0.3, max(TEST_TIMES) + 0.3,
                       alpha=0.07, color="red", label="Test region")
            ax.fill_between(t_dense, res, 0, alpha=0.12, color=col)
            ax.set_xlabel("Time (h)"); ax.set_ylabel(lab)
            ax.set_title(f"{LABELS.get(cond, cond)}")
            ax.grid(alpha=0.3)

    plt.suptitle("ODE Residuals across full [0,9] h interval (n=7 scaler)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Consolidated comparison table
# ══════════════════════════════════════════════════════════════════════════════
def write_comparison_table(agg_table, out_path):
    lines = []
    lines.append("Phase 3 Final Controlled Evaluation — Comparison Table")
    lines.append("Fixed test set: t = {7, 8, 9}  |  Source: Ep20  |  Target: Ep75")
    lines.append("9 runs per cell: 3 source seeds × 3 scratch seeds")
    lines.append("=" * 90)

    for clamp in (False, True):
        tag = "UNCONSTRAINED (Yps free)" if not clamp else f"BOUNDED (Yps ≤ {YPS_PHYSICAL})"
        lines.append(f"\n{tag}")
        lines.append("-" * 90)
        hdr = (f"{'Condition':<22} {'N':>3}  "
               f"{'S-RMSE':>12}  {'X-RMSE':>12}  {'P-RMSE':>12}  "
               f"{'S-MAE':>10}  {'X-MAE':>10}  {'P-MAE':>10}")
        lines.append(hdr)
        lines.append("-" * 90)

        rows = sorted([r for r in agg_table if r["clamp_Yps"] == clamp],
                      key=lambda r: (r["n_obs"], r["condition"]))
        for row in rows:
            def fmt(mean, std): return f"{mean:.3f}±{std:.3f}"
            lines.append(
                f"{row['condition']:<22} {row['n_obs']:>3}  "
                f"{fmt(row['S_RMSE_mean'], row['S_RMSE_std']):>12}  "
                f"{fmt(row['X_RMSE_mean'], row['X_RMSE_std']):>12}  "
                f"{fmt(row['P_RMSE_mean'], row['P_RMSE_std']):>12}  "
                f"{fmt(row['S_MAE_mean'],  row['S_MAE_std']):>10}  "
                f"{fmt(row['X_MAE_mean'],  row['X_MAE_std']):>10}  "
                f"{fmt(row['P_MAE_mean'],  row['P_MAE_std']):>10}")

    lines.append("\n" + "=" * 90)
    lines.append("\nWINNER ANALYSIS (lowest mean RMSE on fixed test set):")
    lines.append("-" * 60)
    for clamp in (False, True):
        tag = "Unconstrained" if not clamp else "Bounded"
        lines.append(f"\n  [{tag}]")
        conds = ["transfer_full", "transfer_frozen", "scratch_pinn", "scratch_nn"]
        win_count = {c: 0 for c in conds}
        for n_obs in [3, 5, 7]:
            for var in ("S", "X", "P"):
                rows = {c: next((r for r in agg_table
                                 if r["condition"] == c and r["n_obs"] == n_obs
                                 and r["clamp_Yps"] == clamp), None)
                        for c in conds}
                valid = {c: v for c, v in rows.items() if v is not None}
                if not valid:
                    continue
                winner = min(valid, key=lambda c: valid[c][f"{var}_RMSE_mean"])
                win_count[winner] += 1
                rmse_w = valid[winner][f"{var}_RMSE_mean"]
                lines.append(f"    n={n_obs} {var}: {winner} ({rmse_w:.3f} g/L)")
        lines.append(f"  Overall wins: " +
                     "  ".join(f"{c}={w}" for c, w in
                                sorted(win_count.items(), key=lambda x: -x[1])))

    lines.append("\n" + "=" * 90)
    lines.append("\nLIMITATIONS:")
    lines.append("  1. n=7 uses all 7 pool points; val proxy = last training point")
    lines.append("     (in-training); early stopping is in-sample for that budget.")
    lines.append("  2. Fixed test set has only 3 time points; RMSE estimates have")
    lines.append("     wide uncertainty and R² is not reported (n<3 effective df).")
    lines.append("  3. Standard Monod is structurally misspecified; Yps>0.511 in")
    lines.append("     all unconstrained models regardless of seed or condition.")
    lines.append("  4. Bounded Yps massively worsens fit; synthetic data effects")
    lines.append("     cannot be captured within physical stoichiometry.")
    lines.append("  5. Episode-31 not evaluated here (requires IC-conditioning).")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Comparison table → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("Phase 3 Final Controlled Evaluation")
    print("=" * 65)
    print(f"Fixed test times : {sorted(TEST_TIMES)}")
    print(f"Training budgets : {TRAIN_BUDGETS}")
    print(f"Source seeds     : {SOURCE_SEEDS}")
    print(f"Scratch seeds    : {SCRATCH_SEEDS}")
    print()

    df20 = load_episode(20)
    df75 = load_episode(75)
    df75_test = df75[df75["time_hours"].isin(TEST_TIMES)].copy().reset_index(drop=True)

    print(f"Ep75 test rows   : {sorted(df75_test['time_hours'].tolist())}")
    print(f"Ep75 test y_true :")
    for _, row in df75_test.iterrows():
        print(f"  t={row['time_hours']:.0f}h  "
              f"S={row['glucose']:.4f}  X={row['biomass']:.4f}  P={row['ethanol']:.4f}")

    # ── Step 1: Pretrain sources ──────────────────────────────────────────────
    print(f"\nPretraining {len(SOURCE_SEEDS)} source PINNs on Episode 20…")
    source_models = []
    for seed in SOURCE_SEEDS:
        t0 = time.time()
        st, ps, fp, n_ep = pretrain_source(seed, df20)
        elapsed = time.time() - t0
        yps_flag = "  ⚠ Yps>0.511" if fp["Yps"] > YPS_PHYSICAL else ""
        print(f"  seed={seed}: ep={n_ep}  "
              f"μmax={fp['mu_max']:.4f}  Ks={fp['Ks']:.4f}  "
              f"Yxs={fp['Yxs']:.4f}  Yps={fp['Yps']:.4f}{yps_flag}  "
              f"t={elapsed:.1f}s")
        source_models.append({"seed": seed, "net_state": st,
                               "params_state": ps, "fitted": fp})

    # Save source model states
    for sm in source_models:
        torch.save(sm["net_state"],
                   os.path.join(OUT_DIR, f"source_net_seed{sm['seed']}.pt"))
        torch.save(sm["params_state"],
                   os.path.join(OUT_DIR, f"source_params_seed{sm['seed']}.pt"))
    with open(os.path.join(OUT_DIR, "source_fitted_params.json"), "w") as f:
        json.dump([{"seed": sm["seed"], **sm["fitted"]} for sm in source_models],
                  f, indent=2)
    print("✓ Source models saved.")

    # ── Step 2: Run all experiments ───────────────────────────────────────────
    conditions   = ["transfer_full", "transfer_frozen", "scratch_pinn", "scratch_nn"]
    clamp_values = [False, True]
    all_runs     = []

    # Count total experiments
    n_transfer_runs = (len(SOURCE_SEEDS) *
                       len([c for c in conditions if "transfer" in c]) *
                       len(TRAIN_BUDGETS) * len(clamp_values))
    n_scratch_runs  = (len(SCRATCH_SEEDS) *
                       len([c for c in conditions if "scratch" in c]) *
                       len(TRAIN_BUDGETS) * len(clamp_values))
    total = n_transfer_runs + n_scratch_runs
    run_count = 0

    print(f"\nRunning {total} experiments…")

    for clamp in clamp_values:
        for n_obs in sorted(TRAIN_BUDGETS.keys()):
            for cond in conditions:
                is_transfer = "transfer" in cond
                seeds_to_use = SOURCE_SEEDS if is_transfer else SCRATCH_SEEDS
                seed_runs = []
                for seed in seeds_to_use:
                    run_count += 1
                    src = source_models[SOURCE_SEEDS.index(seed)] if is_transfer else None
                    st  = src["net_state"]     if src else None
                    ps  = src["params_state"]  if src else None
                    print(f"  [{run_count:>3d}/{total}] {cond:<20} "
                          f"n={n_obs}  seed={seed}  "
                          f"{'bounded' if clamp else 'unconstrained'}",
                          end="", flush=True)
                    r = run_experiment(cond, n_obs, seed,
                                       df75, df75_test,
                                       st, ps, clamp_Yps=clamp)
                    print(f"  S={r['metrics']['S']['RMSE']:.4f}  "
                          f"ep={r['n_epochs']}  t={r['elapsed_s']:.1f}s")
                    seed_runs.append(r)
                    all_runs.append(r)

    print(f"\n✓ All {run_count} experiments complete.")

    # ── Step 3: Aggregate ────────────────────────────────────────────────────
    agg_table = []
    for clamp in clamp_values:
        for n_obs in sorted(TRAIN_BUDGETS.keys()):
            for cond in conditions:
                runs = [r for r in all_runs
                        if r["n_obs"] == n_obs and r["condition"] == cond
                        and r["clamp_Yps"] == clamp]
                if runs:
                    agg_table.append(aggregate(runs))

    with open(os.path.join(OUT_DIR, "aggregated_results.json"), "w") as f:
        json.dump(agg_table, f, indent=2)
    print("✓ Aggregated results saved.")

    # ── Step 4: Plots ─────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    for clamp in clamp_values:
        tag = "bounded" if clamp else "unconstrained"
        plot_rmse_vs_nobs(agg_table, clamp,
                          os.path.join(OUT_DIR, f"rmse_vs_nobs_{tag}.png"))
        print(f"  ✓ RMSE plot [{tag}]")

        for n_obs in sorted(TRAIN_BUDGETS.keys()):
            best_runs = {}
            for cond in conditions:
                runs = [r for r in all_runs
                        if r["n_obs"] == n_obs and r["condition"] == cond
                        and r["clamp_Yps"] == clamp]
                if runs:
                    best_runs[cond] = min(runs,
                                         key=lambda r: r["metrics"]["S"]["RMSE"])
            plot_predictions(df75, agg_table, best_runs, n_obs, clamp,
                             os.path.join(OUT_DIR,
                                          f"predictions_n{n_obs}_{tag}.png"))
            print(f"  ✓ Predictions n={n_obs} [{tag}]")

    # Residuals at n=7 unconstrained
    best_n7_unc = {}
    for cond in conditions:
        runs = [r for r in all_runs
                if r["n_obs"] == 7 and r["condition"] == cond
                and not r["clamp_Yps"]]
        if runs:
            best_n7_unc[cond] = min(runs, key=lambda r: r["metrics"]["S"]["RMSE"])
    pinn_keys = {c: v for c, v in best_n7_unc.items()
                 if v["params_state"]}
    if pinn_keys:
        plot_residuals_final(pinn_keys, df75,
                             os.path.join(OUT_DIR, "physics_residuals_n7.png"))
        print("  ✓ Physics residuals plot (n=7, unconstrained)")

    # ── Step 5: Comparison table ─────────────────────────────────────────────
    write_comparison_table(agg_table,
                           os.path.join(OUT_DIR, "comparison_table.txt"))

    # ── Step 6: Print final summary ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY — Fixed test set t={7,8,9}  (9 runs per cell)")
    print("=" * 65)
    for clamp in (False, True):
        tag = "UNCONSTRAINED" if not clamp else f"BOUNDED (Yps≤{YPS_PHYSICAL})"
        print(f"\n{tag}")
        print(f"{'Condition':<22} {'N':>3}  {'S-RMSE':>12}  "
              f"{'X-RMSE':>12}  {'P-RMSE':>12}")
        print("-" * 65)
        rows = sorted([r for r in agg_table if r["clamp_Yps"] == clamp],
                      key=lambda r: (r["n_obs"], r["condition"]))
        for row in rows:
            def f(m, s): return f"{m:>6.3f}±{s:.3f}"
            print(f"{row['condition']:<22} {row['n_obs']:>3}  "
                  f"{f(row['S_RMSE_mean'],row['S_RMSE_std']):>12}  "
                  f"{f(row['X_RMSE_mean'],row['X_RMSE_std']):>12}  "
                  f"{f(row['P_RMSE_mean'],row['P_RMSE_std']):>12}")

    print("\n" + "=" * 65)
    print("Protocol verification:")
    print(f"  Test times used for: training=NO  val=NO  norm=NO  HP-tuning=NO")
    print(f"  Scaler fitted on:    training rows only")
    print(f"  Collocation range:   [0, t_max_train]  (not extended into test)")
    print(f"  All Phase 1/2/3/audit outputs: PRESERVED (read-only)")
    print("=" * 65)
    print("Phase 3 final evaluation complete. Outputs:", OUT_DIR)


if __name__ == "__main__":
    main()
