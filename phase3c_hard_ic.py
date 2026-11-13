"""
Phase 3C: Hard-Initial-Condition Conditional PINN
Transfer Learning with Physics-Informed Neural Networks for Bioethanol Fermentation

IC-enforcing transform (exact, not approximate):
    Let g(t, y0_n) = net([t_norm, y0_n])       # unconstrained MLP output  (N, 3)
    t_frac = t_norm / (t_norm + eps)             # ∈ [0, 1),  0 at t=0
    y_norm(t) = y0_n * exp( clamp(t_frac * g, -30, 30) )

    At t=0  : t_frac=0  →  y_norm = y0_n * exp(0) = y0_n   ← EXACT IC
    For all t: y_norm = y0_n * exp(…) ≥ 0 since y0_n ≥ 0   ← nonneg
    Near-zero y0_n: exp(0)=1, output = y0_n exactly; stays near-zero.

    Physical output: y_phys = y_norm * scale (same as before)

Inputs  : [t_norm, S0_n, X0_n, P0_n]   (4-dim)
Outputs : [S_n, X_n, P_n]               (3-dim normalised, exact IC at t=0)

Multi-source pretraining: joint mini-batch (randomly sample episodes each step).
Unit tests run before any training.
Fixed test set t={7,8,9} for Ep75; nested budgets n=3,5,7 from t=0..6.
Outputs: outputs/phase3c/
"""

import os, sys, json, copy, time, warnings
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
DATA_PATH = os.path.join(BASE_DIR, "data", "complete_fermentation_dataset.csv")
P1_DIR    = os.path.join(BASE_DIR, "outputs", "phase1")
P3F_DIR   = os.path.join(BASE_DIR, "outputs", "phase3_final")
OUT_DIR   = os.path.join(BASE_DIR, "outputs", "phase3c")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Protocol constants ─────────────────────────────────────────────────────────
TEST_TIMES    = {7.0, 8.0, 9.0}
TRAIN_BUDGETS = {
    3: [0.0, 3.0, 6.0],
    5: [0.0, 1.0, 3.0, 5.0, 6.0],
    7: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
SEEDS              = [42, 7, 123]
MULTI_SOURCE_EPS   = [27, 55, 11, 47, 10, 80, 78, 82]
SINGLE_SOURCE_EP   = 20
YPS_MAX            = 0.511
N_COLLOC           = 150
IC_EPS             = 1e-2        # gate denominator: t_frac = t/(t+IC_EPS)
EXP_CLAMP          = 30.0        # prevent overflow in exp

# Fixed hyper-parameters
LR_SCRATCH   = 1e-3
LR_FINETUNE  = 3e-4
MAX_EPOCHS   = 15_000
PATIENCE     = 1_000
LR_P         = 500
LR_F         = 0.5
MIN_LR       = 1e-6
W_DATA       = 1.0
W_PHYS       = 0.1
W_IC         = 0.0       # IC enforced exactly by construction; no soft penalty
KINETIC_INIT = dict(mu_max=0.8, Ks=0.5, Yxs=0.45, Yps=0.50)

# Multi-source joint training hyper-parameters
MULTI_BATCH_SIZE  = 4      # episodes sampled per step
MULTI_MAX_STEPS   = 20_000
MULTI_PATIENCE    = 2_000
MULTI_LR          = 1e-3


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)

def r2(yt, yp):
    ss_res = float(np.sum((yt - yp)**2))
    ss_tot = float(np.sum((yt - yt.mean())**2))
    return float("nan") if ss_tot < 1e-10 else 1.0 - ss_res / ss_tot

def compute_metrics(y_true, y_pred):
    out = {}
    for i, lab in enumerate(("S", "X", "P")):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[lab] = {"RMSE": float(np.sqrt(np.mean((yt-yp)**2))),
                    "MAE":  float(np.mean(np.abs(yt-yp))),
                    "R2":   r2(yt, yp) if len(yt) >= 3 else float("nan")}
    return out

def load_episode_df(ep_id):
    return (pd.read_csv(os.path.join(P1_DIR, f"episode_{ep_id}_cleaned.csv"))
              .sort_values("time_hours").reset_index(drop=True))

def load_raw_episode(ep_id, df_raw):
    ep = df_raw[df_raw["episode_id"] == ep_id].sort_values("time_hours").reset_index(drop=True)
    below = ep[ep["glucose"] < 0.5]
    if not below.empty:
        ep = ep.loc[:below.index[0]].copy()
    return ep


# ══════════════════════════════════════════════════════════════════════════════
# Scaler — fitted on training data only
# ══════════════════════════════════════════════════════════════════════════════
class Scaler:
    def __init__(self, df_train, t_max_full=None):
        self.t_min = float(df_train["time_hours"].min())
        self.t_max = float(df_train["time_hours"].max())
        self.t_max_full = t_max_full if t_max_full is not None else self.t_max
        self.s_max = float(df_train["glucose"].max())
        self.x_max = float(df_train["biomass"].max())
        self.p_max = float(df_train["ethanol"].max())

    def norm_t(self, t):
        return (np.asarray(t, float) - self.t_min) / (self.t_max - self.t_min + 1e-8)

    def norm_s(self, s):  return np.asarray(s, float) / (self.s_max + 1e-8)
    def norm_x(self, x):  return np.asarray(x, float) / (self.x_max + 1e-8)
    def norm_p(self, p):  return np.asarray(p, float) / (self.p_max + 1e-8)

    def ic_vec(self, S0, X0, P0):
        return torch.tensor(
            [self.norm_s(S0), self.norm_x(X0), self.norm_p(P0)],
            dtype=torch.float32)

    def tensors(self, df, S0, X0, P0):
        t  = torch.tensor(self.norm_t(df["time_hours"].values),
                          dtype=torch.float32).unsqueeze(1)
        s  = torch.tensor(self.norm_s(df["glucose"].values),
                          dtype=torch.float32).unsqueeze(1)
        x  = torch.tensor(self.norm_x(df["biomass"].values),
                          dtype=torch.float32).unsqueeze(1)
        p  = torch.tensor(self.norm_p(df["ethanol"].values),
                          dtype=torch.float32).unsqueeze(1)
        ic = self.ic_vec(S0, X0, P0)
        return t, s, x, p, ic

    def denorm(self, arr):
        return np.column_stack([
            arr[:, 0] * self.s_max,
            arr[:, 1] * self.x_max,
            arr[:, 2] * self.p_max])


# ══════════════════════════════════════════════════════════════════════════════
# Hard-IC Model
# ══════════════════════════════════════════════════════════════════════════════
class HardICMLP(nn.Module):
    """
    Exact initial-condition enforcement via exponential residual transform.

    Forward:
        x_in   = [t_norm, S0_n, X0_n, P0_n]        shape (N, 4)
        g      = net(x_in)                           shape (N, 3)  unconstrained
        t_frac = t_norm / (t_norm + IC_EPS)          shape (N, 1)  ∈ [0,1)
        y_n    = y0_n * exp( clamp(t_frac * g, -EXP_CLAMP, EXP_CLAMP) )

    At t=0: t_frac=0 → y_n = y0_n * exp(0) = y0_n  (EXACT)
    y_n ≥ 0 always (y0_n ≥ 0, exp(.) > 0)

    Near-zero y0_n: output stays near-zero, no numerical issues.
    Exactly-zero y0_n: output = 0 for all t (correct; not applicable here
        since all our ICs have positive concentrations).
    """
    def __init__(self, hidden=32, n_layers=3):
        super().__init__()
        layers = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))   # unconstrained output
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t_norm, ic_norm):
        """
        t_norm  : (N,1)   normalised time
        ic_norm : (3,) or (N,3)  [S0_n, X0_n, P0_n]
        Returns : (N,3) normalised concentrations, nonneg, exact IC at t=0
        """
        N = t_norm.shape[0]
        if ic_norm.dim() == 1:
            ic_norm = ic_norm.unsqueeze(0).expand(N, -1)   # (N,3)
        x_in   = torch.cat([t_norm, ic_norm], dim=1)       # (N,4)
        g      = self.net(x_in)                             # (N,3)
        t_frac = t_norm / (t_norm + IC_EPS)                # (N,1)
        exponent = torch.clamp(t_frac * g, -EXP_CLAMP, EXP_CLAMP)
        return ic_norm * torch.exp(exponent)                # (N,3)  ≥ 0


class KineticParams(nn.Module):
    def __init__(self, **init):
        super().__init__()
        def inv(x): return float(np.log(np.exp(float(x)) - 1.0 + 1e-8))
        self._mu_max = nn.Parameter(torch.tensor(inv(init.get("mu_max", 0.8))))
        self._Ks     = nn.Parameter(torch.tensor(inv(init.get("Ks",     0.5))))
        self._Yxs    = nn.Parameter(torch.tensor(inv(init.get("Yxs",    0.45))))
        self._Yps    = nn.Parameter(torch.tensor(inv(init.get("Yps",    0.50))))

    @property
    def mu_max(self): return nn.functional.softplus(self._mu_max)
    @property
    def Ks(self):     return nn.functional.softplus(self._Ks)
    @property
    def Yxs(self):    return nn.functional.softplus(self._Yxs)
    @property
    def Yps(self):    return nn.functional.softplus(self._Yps)
    def to_dict(self):
        return {k: getattr(self, k).item() for k in ("mu_max","Ks","Yxs","Yps")}


# ══════════════════════════════════════════════════════════════════════════════
# Unit Tests  (Step 2)
# ══════════════════════════════════════════════════════════════════════════════
def run_unit_tests():
    print("\n" + "=" * 55)
    print("UNIT TESTS: Hard-IC enforcement")
    print("=" * 55)

    passed = 0; failed = 0

    # Test several random IC vectors
    test_ics = [
        torch.tensor([1.0,  0.08, 0.03]),   # Ep75-like
        torch.tensor([0.9,  0.10, 0.02]),   # Ep20-like
        torch.tensor([0.5,  0.15, 0.10]),   # mid-range
        torch.tensor([0.01, 0.01, 0.001]),  # near-zero
    ]

    net = HardICMLP()   # fresh random weights

    for trial, ic in enumerate(test_ics):
        t0 = torch.zeros(1, 1, dtype=torch.float32)
        with torch.no_grad():
            pred0 = net(t0, ic.unsqueeze(0))[0]   # (3,)

        ic_err_norm = float((pred0 - ic).abs().max())
        ic_phys_err = float((pred0 - ic).abs().max()) * 25.0  # rough g/L scale

        test_pass = ic_err_norm < 1e-5
        tag = "PASS" if test_pass else "FAIL"
        if test_pass: passed += 1
        else: failed += 1
        print(f"  [{tag}] IC trial {trial+1}: ic={ic.numpy().round(4)}  "
              f"pred0={pred0.numpy().round(6)}  "
              f"max_err_norm={ic_err_norm:.2e}  "
              f"~{ic_phys_err:.2e} g/L")

    # Nonneg at various times
    t_test = torch.tensor([[0.],[0.1],[0.5],[1.0]], dtype=torch.float32)
    ic_test = torch.tensor([0.9, 0.08, 0.03])
    with torch.no_grad():
        preds = net(t_test, ic_test)
    min_val = float(preds.min())
    nonneg_ok = min_val >= 0.0
    tag = "PASS" if nonneg_ok else "FAIL"
    if nonneg_ok: passed += 1
    else: failed += 1
    print(f"  [{tag}] Nonnegativity: min_pred={min_val:.6f}")

    # Finite derivatives
    t_grad = torch.tensor([[0.3]], requires_grad=True)
    ic_g   = torch.tensor([0.9, 0.08, 0.03])
    pred_g = net(t_grad, ic_g)
    grad   = torch.autograd.grad(pred_g.sum(), t_grad, create_graph=False)[0]
    finite_ok = bool(torch.isfinite(grad).all())
    tag = "PASS" if finite_ok else "FAIL"
    if finite_ok: passed += 1
    else: failed += 1
    print(f"  [{tag}] Finite derivatives at t=0.3: grad={float(grad):.4f}")

    # IC error in physical g/L units (scale = 25 g/L as rough proxy)
    SCALE_GL = 25.0
    for trial, ic in enumerate(test_ics[:3]):
        t0 = torch.zeros(1,1,dtype=torch.float32)
        with torch.no_grad():
            pred0 = net(t0, ic.unsqueeze(0))[0]
        err_gl = float((pred0 - ic).abs().max()) * SCALE_GL
        ic_ok  = err_gl < 1e-6
        tag = "PASS" if ic_ok else "FAIL"
        if ic_ok: passed += 1
        else: failed += 1
        print(f"  [{tag}] IC physical err trial {trial+1}: {err_gl:.2e} g/L  (<1e-6?)")

    print(f"\n  Results: {passed} passed, {failed} failed")
    if failed > 0:
        print(f"  ✗ {failed} UNIT TEST(S) FAILED — aborting training.")
        sys.exit(1)
    else:
        print("  ✓ All unit tests passed. Proceeding to training.")
    return passed, failed


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════
_mse = nn.MSELoss()

def loss_data(model, t, s, x, p, ic):
    pred = model(t, ic)
    return _mse(pred[:,0:1], s) + _mse(pred[:,1:2], x) + _mse(pred[:,2:3], p)

def loss_phys(model, params, scaler, tc, ic):
    tc2  = tc.clone().requires_grad_(True)
    pred = model(tc2, ic)
    Sn, Xn, Pn = pred[:,0:1], pred[:,1:2], pred[:,2:3]
    o    = torch.ones_like(Sn)
    dSn  = torch.autograd.grad(Sn, tc2, o, create_graph=True)[0]
    dXn  = torch.autograd.grad(Xn, tc2, o, create_graph=True, retain_graph=True)[0]
    dPn  = torch.autograd.grad(Pn, tc2, o, create_graph=True, retain_graph=True)[0]
    T    = scaler.t_max - scaler.t_min
    dS = dSn*(scaler.s_max/T); dX = dXn*(scaler.x_max/T); dP = dPn*(scaler.p_max/T)
    S  = Sn*scaler.s_max;      X  = Xn*scaler.x_max
    mu = params.mu_max * S / (params.Ks + S + 1e-8)
    rX = dX - mu*X
    rS = dS + (1/params.Yxs)*mu*X
    rP = dP - params.Yps*(1/params.Yxs)*mu*X
    return (rX.pow(2).mean() + rS.pow(2).mean() + rP.pow(2).mean()) / 3.0

def eval_residuals(model, params, scaler, ic, n=150):
    t_p = np.linspace(scaler.t_min, scaler.t_max, n)
    tc  = torch.tensor(scaler.norm_t(t_p), dtype=torch.float32,
                       requires_grad=True).unsqueeze(1)
    pred = model(tc, ic)
    Sn, Xn, Pn = pred[:,0:1], pred[:,1:2], pred[:,2:3]
    dSn = torch.autograd.grad(Sn, tc, torch.ones_like(Sn), retain_graph=True)[0]
    dXn = torch.autograd.grad(Xn, tc, torch.ones_like(Xn), retain_graph=True)[0]
    dPn = torch.autograd.grad(Pn, tc, torch.ones_like(Pn), retain_graph=False)[0]
    T   = scaler.t_max - scaler.t_min
    dS  = dSn.detach().numpy()[:,0]*scaler.s_max/T
    dX  = dXn.detach().numpy()[:,0]*scaler.x_max/T
    dP  = dPn.detach().numpy()[:,0]*scaler.p_max/T
    S   = Sn.detach().numpy()[:,0]*scaler.s_max
    X   = Xn.detach().numpy()[:,0]*scaler.x_max
    mu  = params.mu_max.item()*S/(params.Ks.item()+S+1e-8)
    rX  = dX - mu*X
    rS  = dS + (1/params.Yxs.item())*mu*X
    rP  = dP - params.Yps.item()*(1/params.Yxs.item())*mu*X
    return {"rX_rms": float(np.sqrt(np.mean(rX**2))),
            "rS_rms": float(np.sqrt(np.mean(rS**2))),
            "rP_rms": float(np.sqrt(np.mean(rP**2)))}


# ══════════════════════════════════════════════════════════════════════════════
# Single-episode training loop
# ══════════════════════════════════════════════════════════════════════════════
def train_single_ep(model, params, scaler, df_train, df_val,
                    S0, X0, P0, lr=LR_SCRATCH, frozen_layer_0=False):
    ic = scaler.ic_vec(S0, X0, P0)
    t_obs, s_obs, x_obs, p_obs, _ = scaler.tensors(df_train, S0, X0, P0)
    t_val, s_val, x_val, p_val, _ = scaler.tensors(df_val,   S0, X0, P0)
    tc = torch.tensor(
        scaler.norm_t(np.linspace(scaler.t_min, scaler.t_max, N_COLLOC)),
        dtype=torch.float32).unsqueeze(1)

    if frozen_layer_0:
        first = [m for m in model.modules() if isinstance(m, nn.Linear)][0]
        for p in first.parameters(): p.requires_grad_(False)

    trainable = ([p for p in model.parameters() if p.requires_grad] +
                 list(params.parameters()))
    opt   = torch.optim.Adam(trainable, lr=lr)
    sched = ReduceLROnPlateau(opt, "min", factor=LR_F,
                               patience=LR_P, min_lr=MIN_LR)
    best_val, best_state, patience_ct = float("inf"), None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        ld = loss_data(model, t_obs, s_obs, x_obs, p_obs, ic)
        lp = loss_phys(model, params, scaler, tc, ic)
        total = W_DATA*ld + W_PHYS*lp
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(t_val, ic)
            lv = _mse(pv[:,0:1],s_val)+_mse(pv[:,1:2],x_val)+_mse(pv[:,2:3],p_val)
        sched.step(lv.item())
        if lv.item() < best_val - 1e-7:
            best_val, patience_ct = lv.item(), 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break

    if best_state: model.load_state_dict(best_state)
    for p in model.parameters(): p.requires_grad_(True)
    return epoch+1, best_val


# ══════════════════════════════════════════════════════════════════════════════
# Joint multi-source training  (Step 3)
# ══════════════════════════════════════════════════════════════════════════════
def build_source_datasets(ep_ids, df_raw):
    """Build a list of (df, S0, X0, P0, t_max) per episode."""
    datasets = []
    for ep_id in ep_ids:
        df = load_raw_episode(ep_id, df_raw)
        if len(df) < 4:
            continue
        S0 = float(df["glucose"].iloc[0])
        X0 = float(df["biomass"].iloc[0])
        P0 = float(df["ethanol"].iloc[0])
        datasets.append({"ep_id": ep_id, "df": df, "S0": S0, "X0": X0, "P0": P0,
                          "t_max": float(df["time_hours"].max())})
    return datasets


def train_joint_multi(model, params, datasets, seed):
    """
    Joint mini-batch training across multiple source episodes.
    Each step randomly draws MULTI_BATCH_SIZE episodes and accumulates
    their data + physics losses before a single gradient step.
    A shared global scaler (max over all sources) is used.
    """
    # Build global scaler from all source data
    all_dfs = pd.concat([d["df"] for d in datasets], ignore_index=True)
    global_s_max = float(all_dfs["glucose"].max())
    global_x_max = float(all_dfs["biomass"].max())
    global_p_max = float(all_dfs["ethanol"].max())
    global_t_max = float(all_dfs["time_hours"].max())

    class GlobalScaler:
        t_min = 0.0
        t_max = global_t_max
        s_max = global_s_max
        x_max = global_x_max
        p_max = global_p_max

        @staticmethod
        def norm_t(t):
            return (np.asarray(t, float)) / (GlobalScaler.t_max + 1e-8)
        @staticmethod
        def norm_s(s): return np.asarray(s, float) / (GlobalScaler.s_max + 1e-8)
        @staticmethod
        def norm_x(x): return np.asarray(x, float) / (GlobalScaler.x_max + 1e-8)
        @staticmethod
        def norm_p(p): return np.asarray(p, float) / (GlobalScaler.p_max + 1e-8)
        @staticmethod
        def ic_vec(S0, X0, P0):
            return torch.tensor(
                [GlobalScaler.norm_s(S0),
                 GlobalScaler.norm_x(X0),
                 GlobalScaler.norm_p(P0)], dtype=torch.float32)

    gsc = GlobalScaler()

    # Pre-compute tensors for every source episode
    ep_tensors = []
    for d in datasets:
        df    = d["df"]
        S0, X0, P0 = d["S0"], d["X0"], d["P0"]
        t_ep  = float(df["time_hours"].max())
        t_obs = torch.tensor(gsc.norm_t(df["time_hours"].values),
                             dtype=torch.float32).unsqueeze(1)
        s_obs = torch.tensor(gsc.norm_s(df["glucose"].values),
                             dtype=torch.float32).unsqueeze(1)
        x_obs = torch.tensor(gsc.norm_x(df["biomass"].values),
                             dtype=torch.float32).unsqueeze(1)
        p_obs = torch.tensor(gsc.norm_p(df["ethanol"].values),
                             dtype=torch.float32).unsqueeze(1)
        ic    = gsc.ic_vec(S0, X0, P0)
        tc    = torch.tensor(
            gsc.norm_t(np.linspace(0, t_ep, N_COLLOC)),
            dtype=torch.float32).unsqueeze(1)
        ep_tensors.append({"ep_id": d["ep_id"], "t_obs": t_obs,
                            "s_obs": s_obs, "x_obs": x_obs, "p_obs": p_obs,
                            "ic": ic, "tc": tc,
                            "S0": S0, "X0": X0, "P0": P0})

    n_ep = len(ep_tensors)
    rng  = np.random.default_rng(seed)
    trainable = list(model.parameters()) + list(params.parameters())
    opt  = torch.optim.Adam(trainable, lr=MULTI_LR)
    sched = ReduceLROnPlateau(opt, "min", factor=LR_F,
                               patience=500, min_lr=MIN_LR)

    best_loss, best_state, best_params_state = float("inf"), None, None
    patience_ct = 0
    per_ep_losses = {d["ep_id"]: [] for d in datasets}

    for step in range(MULTI_MAX_STEPS):
        model.train()
        opt.zero_grad()
        # Sample a batch of episodes
        batch_idx = rng.choice(n_ep, size=min(MULTI_BATCH_SIZE, n_ep), replace=False)
        total_loss = torch.tensor(0.0)
        for bi in batch_idx:
            ep = ep_tensors[bi]
            ic = ep["ic"]
            ld = loss_data(model, ep["t_obs"], ep["s_obs"],
                            ep["x_obs"], ep["p_obs"], ic)

            # Use a local scaler-like object for physics
            class _LocalScaler:
                t_min = 0.0
                t_max = GlobalScaler.t_max
                s_max = GlobalScaler.s_max
                x_max = GlobalScaler.x_max
                p_max = GlobalScaler.p_max

            lp = loss_phys(model, params, _LocalScaler(), ep["tc"], ic)
            total_loss = total_loss + W_DATA*ld + W_PHYS*lp

        total_loss = total_loss / len(batch_idx)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        # Track mean loss across all episodes for patience
        model.eval()
        with torch.no_grad():
            mean_loss = sum(
                (loss_data(model, ep["t_obs"], ep["s_obs"],
                            ep["x_obs"], ep["p_obs"], ep["ic"])).item()
                for ep in ep_tensors) / n_ep

        sched.step(mean_loss)
        if mean_loss < best_loss - 1e-7:
            best_loss, patience_ct = mean_loss, 0
            best_state       = copy.deepcopy(model.state_dict())
            best_params_state = copy.deepcopy(params.state_dict())
        else:
            patience_ct += 1
            if patience_ct >= MULTI_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
        params.load_state_dict(best_params_state)

    # Per-episode final losses
    model.eval()
    final_ep_losses = {}
    with torch.no_grad():
        for ep in ep_tensors:
            ld = loss_data(model, ep["t_obs"], ep["s_obs"],
                            ep["x_obs"], ep["p_obs"], ep["ic"])
            final_ep_losses[ep["ep_id"]] = float(ld.item())

    for p in model.parameters(): p.requires_grad_(True)
    return step+1, best_loss, final_ep_losses, gsc


# ══════════════════════════════════════════════════════════════════════════════
# Pretraining helpers
# ══════════════════════════════════════════════════════════════════════════════
def pretrain_single(seed, ep_id):
    set_seed(seed)
    df   = load_episode_df(ep_id)
    S0, X0, P0 = (float(df["glucose"].iloc[0]),
                  float(df["biomass"].iloc[0]),
                  float(df["ethanol"].iloc[0]))
    sc   = Scaler(df.iloc[:-2], t_max_full=float(df["time_hours"].max()))
    net  = HardICMLP()
    par  = KineticParams(**KINETIC_INIT)
    n_ep, bv = train_single_ep(net, par, sc, df.iloc[:-2], df.iloc[-2:],
                                S0, X0, P0, lr=LR_SCRATCH)
    return (copy.deepcopy(net.state_dict()),
            copy.deepcopy(par.state_dict()),
            par.to_dict(), n_ep)


def pretrain_multi_joint(seed, ep_ids, df_raw):
    set_seed(seed)
    datasets = build_source_datasets(ep_ids, df_raw)
    net  = HardICMLP()
    par  = KineticParams(**KINETIC_INIT)
    n_steps, best_loss, ep_losses, gsc = train_joint_multi(net, par, datasets, seed)
    return (copy.deepcopy(net.state_dict()),
            copy.deepcopy(par.state_dict()),
            par.to_dict(), n_steps, best_loss, ep_losses)


# ══════════════════════════════════════════════════════════════════════════════
# IC error measurement in physical g/L
# ══════════════════════════════════════════════════════════════════════════════
def measure_ic_error_gl(model, scaler, S0, X0, P0):
    ic = scaler.ic_vec(S0, X0, P0)
    t0 = torch.zeros(1, 1, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred0 = model(t0, ic.unsqueeze(0))[0].numpy()
    pred_phys = np.array([pred0[0]*scaler.s_max,
                           pred0[1]*scaler.x_max,
                           pred0[2]*scaler.p_max])
    true_phys = np.array([S0, X0, P0])
    return float(np.sqrt(np.mean((pred_phys - true_phys)**2)))


# ══════════════════════════════════════════════════════════════════════════════
# Target experiment
# ══════════════════════════════════════════════════════════════════════════════
def predict_phys(model, scaler, t_arr, S0, X0, P0):
    model.eval()
    ic = scaler.ic_vec(S0, X0, P0)
    tn = torch.tensor(scaler.norm_t(np.asarray(t_arr, float)),
                      dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pn = model(tn, ic).numpy()
    return scaler.denorm(pn)


def run_experiment(condition, n_obs, seed, df75, df75_test,
                   src_single, src_multi):
    set_seed(seed)
    train_times = TRAIN_BUDGETS[n_obs]
    df_train = df75[df75["time_hours"].isin(train_times)].copy().reset_index(drop=True)
    pool = set(t for t in df75["time_hours"] if t not in TEST_TIMES)
    leftover = sorted(pool - set(train_times))
    if leftover:
        mid = leftover[len(leftover)//2]
        df_val = df75[df75["time_hours"] == mid].copy().reset_index(drop=True)
    else:
        df_val = df_train.iloc[[-1]].copy().reset_index(drop=True)

    assert not (set(df_val["time_hours"]) & TEST_TIMES)
    assert not (set(df_train["time_hours"]) & TEST_TIMES)

    S0 = float(df75["glucose"].iloc[0])
    X0 = float(df75["biomass"].iloc[0])
    P0 = float(df75["ethanol"].iloc[0])
    sc = Scaler(df_train, t_max_full=max(TEST_TIMES))

    net = HardICMLP()
    par = KineticParams(**KINETIC_INIT)
    lr  = LR_SCRATCH
    frozen = False

    if condition == "hard_transfer_single":
        net.load_state_dict(copy.deepcopy(src_single["net_state"]))
        par.load_state_dict(copy.deepcopy(src_single["par_state"]))
        lr = LR_FINETUNE
    elif condition == "hard_transfer_single_frozen":
        net.load_state_dict(copy.deepcopy(src_single["net_state"]))
        par.load_state_dict(copy.deepcopy(src_single["par_state"]))
        lr = LR_FINETUNE; frozen = True
    elif condition == "hard_transfer_multi":
        net.load_state_dict(copy.deepcopy(src_multi["net_state"]))
        par.load_state_dict(copy.deepcopy(src_multi["par_state"]))
        lr = LR_FINETUNE

    t_start = time.time()
    n_ep, bv = train_single_ep(net, par, sc, df_train, df_val,
                                S0, X0, P0, lr=lr, frozen_layer_0=frozen)
    elapsed = time.time() - t_start

    # IC error in physical g/L
    ic_err_gl = measure_ic_error_gl(net, sc, S0, X0, P0)

    y_test_true = df75_test[["glucose","biomass","ethanol"]].values
    y_test_pred = predict_phys(net, sc, df75_test["time_hours"].values, S0, X0, P0)
    m = compute_metrics(y_test_true, y_test_pred)

    ic = sc.ic_vec(S0, X0, P0)
    try:
        res = eval_residuals(net, par, sc, ic)
    except Exception:
        res = {k: float("nan") for k in ("rX_rms","rS_rms","rP_rms")}

    val_note = ("in-training proxy" if not leftover
                else f"pool holdout t={df_val['time_hours'].iloc[0]:.0f}")
    return {
        "condition":   condition,
        "n_obs":       n_obs,
        "seed":        seed,
        "train_times": train_times,
        "val_note":    val_note,
        "metrics":     m,
        "residuals":   res,
        "params":      par.to_dict(),
        "ic_err_gl":   ic_err_gl,
        "n_epochs":    n_ep,
        "elapsed_s":   elapsed,
        "model_state": copy.deepcopy(net.state_dict()),
        "par_state":   copy.deepcopy(par.state_dict()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate
# ══════════════════════════════════════════════════════════════════════════════
def aggregate(runs):
    out = {"condition": runs[0]["condition"],
           "n_obs":     runs[0]["n_obs"],
           "n_runs":    len(runs)}
    for var in ("S","X","P"):
        for met in ("RMSE","MAE"):
            v = np.array([r["metrics"][var][met] for r in runs], float)
            out[f"{var}_{met}_mean"] = float(np.nanmean(v))
            out[f"{var}_{met}_std"]  = float(np.nanstd(v))
    for rk in ("rX_rms","rS_rms","rP_rms"):
        v = np.array([r["residuals"][rk] for r in runs], float)
        out[f"{rk}_mean"] = float(np.nanmean(v))
        out[f"{rk}_std"]  = float(np.nanstd(v))
    out["ic_err_gl_mean"] = float(np.mean([r["ic_err_gl"] for r in runs]))
    out["n_epochs_mean"]  = float(np.mean([r["n_epochs"] for r in runs]))
    out["elapsed_s_mean"] = float(np.mean([r["elapsed_s"] for r in runs]))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════
COND_COLORS = {
    "hard_scratch":                "#E91E63",
    "hard_transfer_single":        "#2196F3",
    "hard_transfer_single_frozen": "#9C27B0",
    "hard_transfer_multi":         "#4CAF50",
}
COND_LABELS = {
    "hard_scratch":                "HardIC Scratch",
    "hard_transfer_single":        "HardIC Transfer (single src)",
    "hard_transfer_single_frozen": "HardIC Transfer (single, frozen)",
    "hard_transfer_multi":         "HardIC Transfer (joint multi)",
}


def plot_rmse_vs_nobs(agg_table, timeonly_agg, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, var in zip(axes, ("S","X","P")):
        # HardIC conditions
        for cond, col in COND_COLORS.items():
            rows = sorted([r for r in agg_table if r["condition"]==cond],
                          key=lambda r: r["n_obs"])
            if not rows: continue
            xs   = [r["n_obs"] for r in rows]
            ys   = [r[f"{var}_RMSE_mean"] for r in rows]
            errs = [r[f"{var}_RMSE_std"]  for r in rows]
            ax.errorbar(xs, ys, yerr=errs, marker="o", lw=1.8,
                        color=col, label=COND_LABELS[cond], capsize=3)
        # Time-only scratch PINN reference from phase3_final
        if timeonly_agg:
            sp = sorted([r for r in timeonly_agg
                         if r["condition"]=="scratch_pinn" and not r.get("clamp_Yps",False)],
                        key=lambda r: r["n_obs"])
            if sp:
                xs2  = [r["n_obs"] for r in sp]
                ys2  = [r[f"{var}_RMSE_mean"] for r in sp]
                errs2= [r[f"{var}_RMSE_std"]  for r in sp]
                ax.errorbar(xs2, ys2, yerr=errs2, marker="s", lw=1.8,
                            linestyle="--", color="#607D8B",
                            label="Ref: time-only scratch PINN", capsize=3)
        ax.set_xlabel("Target observations (pool t=0..6)")
        ax.set_ylabel(f"RMSE {var} (g/L)")
        ax.set_title(f"{var} — Fixed test t={{7,8,9}}")
        ax.set_xticks([3,5,7])
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    plt.suptitle("Phase 3C: Hard-IC PINN — RMSE vs Observations",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(df75, all_runs, n_obs, out_path):
    df_tr = df75[df75["time_hours"].isin(TRAIN_BUDGETS[n_obs])]
    S0 = float(df75["glucose"].iloc[0])
    X0 = float(df75["biomass"].iloc[0])
    P0 = float(df75["ethanol"].iloc[0])
    sc = Scaler(df_tr, t_max_full=9.0)
    t_dense = np.linspace(0.0, 9.0, 300)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, idx) in zip(axes, [("glucose",0),("biomass",1),("ethanol",2)]):
        ax.scatter(df75["time_hours"], df75[col], color="black", s=40, zorder=6,
                   label="All obs (Ep75)")
        ax.scatter(df_tr["time_hours"], df_tr[col], color="lime", s=80,
                   zorder=7, marker="*", label=f"Train ({n_obs} pts)")
        ax.axvspan(min(TEST_TIMES)-0.3, max(TEST_TIMES)+0.3,
                   alpha=0.06, color="red", label="Test region")
        for cond, color in COND_COLORS.items():
            runs = [r for r in all_runs
                    if r["n_obs"]==n_obs and r["condition"]==cond]
            if not runs: continue
            best = min(runs, key=lambda r: r["metrics"]["S"]["RMSE"])
            net = HardICMLP()
            net.load_state_dict(best["model_state"])
            pred = predict_phys(net, sc, t_dense, S0, X0, P0)
            ax.plot(t_dense, pred[:,idx], color=color, lw=1.6,
                    label=COND_LABELS[cond])
        ax.set_xlabel("Time (h)"); ax.set_ylabel(col.capitalize())
        ax.set_title(col.capitalize())
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    plt.suptitle(f"Phase 3C: Hard-IC Predictions — n={n_obs} obs",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════
def write_report(agg_table, src_single_info, src_multi_info,
                 multi_ep_losses, timeonly_agg, out_path):
    lines = []
    lines.append("Phase 3C: Hard-IC Conditional PINN Report")
    lines.append("=" * 80)
    lines.append(f"\nIC transform: y(t) = y0 * exp(clamp(t_frac * net(t,y0), -30, 30))")
    lines.append(f"              t_frac = t / (t + {IC_EPS})")
    lines.append(f"              IC error at t=0: EXACT (y(0) = y0 by construction)")
    lines.append(f"\nFixed test: t={{7,8,9}} | Training pool: t=0..6 | Seeds: {SEEDS}")
    lines.append(f"\nSOURCE KINETIC PARAMETERS")
    lines.append("  Single-source (Ep20):")
    for r in src_single_info:
        flag = "  ⚠ Yps>0.511" if r.get("Yps",0)>YPS_MAX else ""
        lines.append(f"    seed={r['seed']}: μmax={r['mu_max']:.4f}  "
                     f"Ks={r['Ks']:.4f}  Yxs={r['Yxs']:.4f}  "
                     f"Yps={r['Yps']:.4f}{flag}")
    lines.append("  Multi-source joint (8 episodes):")
    for r in src_multi_info:
        flag = "  ⚠ Yps>0.511" if r.get("Yps",0)>YPS_MAX else ""
        lines.append(f"    seed={r['seed']}: μmax={r['mu_max']:.4f}  "
                     f"Ks={r['Ks']:.4f}  Yxs={r['Yxs']:.4f}  "
                     f"Yps={r['Yps']:.4f}{flag}")
    lines.append("\n  Per-source-episode final data losses (multi-source, seed=42):")
    for ep_id, loss in sorted(multi_ep_losses.items()):
        lines.append(f"    Ep{ep_id}: {loss:.5f}")

    lines.append(f"\nRESULTS — Fixed test set (mean ± std, {len(SEEDS)} seeds)")
    lines.append("-" * 80)
    hdr = (f"{'Condition':<34} {'N':>3}  {'S-RMSE':>12}  "
           f"{'X-RMSE':>12}  {'P-RMSE':>12}  {'IC-err(g/L)':>12}")
    lines.append(hdr); lines.append("-"*80)
    for row in sorted(agg_table, key=lambda r:(r["n_obs"],r["condition"])):
        def f(m,s): return f"{m:.3f}±{s:.3f}"
        lines.append(
            f"{row['condition']:<34} {row['n_obs']:>3}  "
            f"{f(row['S_RMSE_mean'],row['S_RMSE_std']):>12}  "
            f"{f(row['X_RMSE_mean'],row['X_RMSE_std']):>12}  "
            f"{f(row['P_RMSE_mean'],row['P_RMSE_std']):>12}  "
            f"{row['ic_err_gl_mean']:>12.2e}")

    lines.append(f"\nCOMPARISON vs Phase 3B (improvement = 3B_S-RMSE - 3C_S-RMSE):")
    # Phase3B scratch was ~5.3 g/L S-RMSE at n=3; Phase3C scratch compared
    lines.append("  (Phase 3B cond_scratch n=3: S≈5.27, Phase 3C hard_scratch n=3: see above)")
    lines.append("  IC error: Phase 3B = 0.562 normalised  ≈ 16.2 g/L")
    lines.append("            Phase 3C = see ic_err_gl column above (target: < 1e-6 g/L)")

    lines.append(f"\nLIMITATIONS:")
    lines.append("  1. n=7 val proxy is in-training; early stopping is in-sample.")
    lines.append("  2. Fixed test set has 3 points; RMSE has wide uncertainty.")
    lines.append("  3. All PINN models produce Yps > 0.511 on standard Monod.")
    lines.append("  4. Joint multi-source training uses global concentration scales;")
    lines.append("     episodes with very different ranges may be down-weighted.")
    lines.append("  5. Hard-IC transform ensures y(0)=y0 but cannot enforce")
    lines.append("     monotone glucose depletion; the network may still produce")
    lines.append("     non-physical trajectories between time points.")
    lines.append("=" * 80)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Report → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("Phase 3C: Hard-IC Conditional PINN")
    print("=" * 65)

    # ── Step 2: Unit tests FIRST ──────────────────────────────────────────────
    run_unit_tests()

    print(f"\nFixed test  : {sorted(TEST_TIMES)}")
    print(f"Train pool  : {TRAIN_BUDGETS}")
    print(f"Single src  : Ep{SINGLE_SOURCE_EP}")
    print(f"Multi src   : {MULTI_SOURCE_EPS}")
    print(f"Seeds       : {SEEDS}")

    df75      = load_episode_df(75)
    df75_test = df75[df75["time_hours"].isin(TEST_TIMES)].copy().reset_index(drop=True)
    df_raw    = pd.read_csv(DATA_PATH)

    # ── Step 3: Pretrain sources ───────────────────────────────────────────────
    print(f"\nPretraining single-source (Ep{SINGLE_SOURCE_EP}) × {len(SEEDS)} seeds…")
    src_single_list = []
    src_single_info = []
    for seed in SEEDS:
        t0 = time.time()
        st, ps, fp, n_ep = pretrain_single(seed, SINGLE_SOURCE_EP)
        dt = time.time()-t0
        flag = "  ⚠ Yps>0.511" if fp["Yps"]>YPS_MAX else ""
        print(f"  seed={seed}: ep={n_ep}  Yps={fp['Yps']:.4f}{flag}  t={dt:.1f}s")
        src_single_list.append({"net_state": st, "par_state": ps, **fp})
        src_single_info.append({"seed": seed, **fp})

    print(f"\nJoint multi-source pretraining ({len(MULTI_SOURCE_EPS)} eps) × {len(SEEDS)} seeds…")
    src_multi_list = []
    src_multi_info = []
    ep_losses_seed0 = {}
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        st, ps, fp, n_steps, bl, ep_losses = pretrain_multi_joint(
            seed, MULTI_SOURCE_EPS, df_raw)
        dt = time.time()-t0
        flag = "  ⚠ Yps>0.511" if fp["Yps"]>YPS_MAX else ""
        print(f"  seed={seed}: steps={n_steps}  best_loss={bl:.5f}  "
              f"Yps={fp['Yps']:.4f}{flag}  t={dt:.1f}s")
        for ep_id, loss in sorted(ep_losses.items()):
            print(f"    Ep{ep_id}: {loss:.5f}", end="  ")
        print()
        src_multi_list.append({"net_state": st, "par_state": ps, **fp})
        src_multi_info.append({"seed": seed, **fp})
        if i == 0:
            ep_losses_seed0 = ep_losses

    torch.save(src_single_list[0]["net_state"],
               os.path.join(OUT_DIR, "src_single_net_seed42.pt"))
    torch.save(src_multi_list[0]["net_state"],
               os.path.join(OUT_DIR, "src_multi_net_seed42.pt"))
    with open(os.path.join(OUT_DIR, "source_params.json"), "w") as f:
        json.dump({"single": src_single_info, "multi": src_multi_info}, f, indent=2)
    print("✓ Source models saved.")

    # ── Step 4: Target experiments ─────────────────────────────────────────────
    conditions = [
        "hard_scratch",
        "hard_transfer_single",
        "hard_transfer_single_frozen",
        "hard_transfer_multi",
    ]
    total     = len(conditions) * len(TRAIN_BUDGETS) * len(SEEDS)
    all_runs  = []
    run_count = 0

    print(f"\nRunning {total} target experiments…")
    for n_obs in sorted(TRAIN_BUDGETS):
        for cond in conditions:
            for idx, seed in enumerate(SEEDS):
                run_count += 1
                ss = src_single_list[idx]
                ms = src_multi_list[idx]
                print(f"  [{run_count:>2d}/{total}] {cond:<34} "
                      f"n={n_obs}  seed={seed}", end="", flush=True)
                r = run_experiment(cond, n_obs, seed, df75, df75_test, ss, ms)
                print(f"  S={r['metrics']['S']['RMSE']:.4f}  "
                      f"IC={r['ic_err_gl']:.2e} g/L  "
                      f"ep={r['n_epochs']}  t={r['elapsed_s']:.1f}s")
                all_runs.append(r)
            # Save best
            best = min([r for r in all_runs if r["n_obs"]==n_obs and r["condition"]==cond],
                       key=lambda r: r["metrics"]["S"]["RMSE"])
            torch.save(best["model_state"],
                       os.path.join(OUT_DIR, f"{cond}_n{n_obs}_best.pt"))

    print(f"\n✓ All {run_count} experiments complete.")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    agg_table = []
    for n_obs in sorted(TRAIN_BUDGETS):
        for cond in conditions:
            runs = [r for r in all_runs
                    if r["n_obs"]==n_obs and r["condition"]==cond]
            if runs: agg_table.append(aggregate(runs))

    with open(os.path.join(OUT_DIR, "aggregated_results.json"), "w") as f:
        json.dump(agg_table, f, indent=2)
    print("✓ Results saved.")

    # ── Load phase3_final reference ────────────────────────────────────────────
    try:
        with open(os.path.join(P3F_DIR, "aggregated_results.json")) as f:
            timeonly_agg = json.load(f)
    except Exception:
        timeonly_agg = None

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_rmse_vs_nobs(agg_table, timeonly_agg,
                      os.path.join(OUT_DIR, "rmse_vs_nobs.png"))
    print("  ✓ RMSE vs obs")
    for n_obs in sorted(TRAIN_BUDGETS):
        plot_predictions(df75, all_runs, n_obs,
                         os.path.join(OUT_DIR, f"predictions_n{n_obs}.png"))
        print(f"  ✓ Predictions n={n_obs}")

    # ── Report ─────────────────────────────────────────────────────────────────
    write_report(agg_table, src_single_info, src_multi_info,
                 ep_losses_seed0, timeonly_agg,
                 os.path.join(OUT_DIR, "phase3c_report.txt"))

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL SUMMARY — Phase 3C Hard-IC PINN")
    print(f"Fixed test t={{7,8,9}} | {len(SEEDS)} seeds per cell")
    print("=" * 65)
    print(f"{'Condition':<34} {'N':>3}  {'S-RMSE':>12}  "
          f"{'X-RMSE':>12}  {'P-RMSE':>12}  {'IC-err(g/L)':>12}")
    print("-" * 65)
    for row in sorted(agg_table, key=lambda r:(r["n_obs"],r["condition"])):
        def f(m,s): return f"{m:>6.3f}±{s:.3f}"
        print(f"{row['condition']:<34} {row['n_obs']:>3}  "
              f"{f(row['S_RMSE_mean'],row['S_RMSE_std']):>12}  "
              f"{f(row['X_RMSE_mean'],row['X_RMSE_std']):>12}  "
              f"{f(row['P_RMSE_mean'],row['P_RMSE_std']):>12}  "
              f"{row['ic_err_gl_mean']:>12.2e}")

    if timeonly_agg:
        print(f"\nReference (time-only scratch PINN, phase3_final):")
        sp = [r for r in timeonly_agg
              if r["condition"]=="scratch_pinn" and not r.get("clamp_Yps",False)]
        for r in sorted(sp, key=lambda x: x["n_obs"]):
            print(f"  n={r['n_obs']}  "
                  f"S={r['S_RMSE_mean']:.3f}±{r['S_RMSE_std']:.3f}  "
                  f"X={r['X_RMSE_mean']:.3f}±{r['X_RMSE_std']:.3f}  "
                  f"P={r['P_RMSE_mean']:.3f}±{r['P_RMSE_std']:.3f}")

    print("=" * 65)
    print("Phase 3C complete.  Outputs:", OUT_DIR)


if __name__ == "__main__":
    main()
