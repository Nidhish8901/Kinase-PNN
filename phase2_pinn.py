"""
Phase 2: Physics-Informed Neural Network Development
Transfer Learning with Physics-Informed Neural Networks for Bioethanol Fermentation

Training episode : Episode 20  (source, 9 observations)
Architecture     : MLP  time -> [S, X, P],  3 hidden x 32 neurons, Tanh
Physics          : Standard Monod kinetics (ODE residuals at collocation pts)
Baseline         : Same MLP, data loss only
Outputs          : outputs/phase2/
"""

import os, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
P1_DIR    = os.path.join(BASE_DIR, "outputs", "phase1")
OUT_DIR   = os.path.join(BASE_DIR, "outputs", "phase2")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")   # keeps memory light

# ── Training hyper-parameters ─────────────────────────────────────────────────
N_COLLOC       = 150          # physics collocation points
LAMBDA_DATA    = 1.0
LAMBDA_PHYS    = 0.1
LAMBDA_IC      = 1.0
LR             = 1e-3
MAX_EPOCHS     = 20_000
PATIENCE       = 1_000        # early-stop patience
LR_PATIENCE    = 500
LR_FACTOR      = 0.5
MIN_LR         = 1e-6

# ── Kinetic parameter initialisation (physically plausible) ───────────────────
# μmax ~0.5–1.5 h⁻¹,  Ks ~0.5–2 g/L,  Yxs ~0.3–0.6,  Yps ~0.4–0.6
MU_MAX_INIT  = 0.8
KS_INIT      = 0.5
YXS_INIT     = 0.45
YPS_INIT     = 0.50

# ══════════════════════════════════════════════════════════════════════════════
# Data loading & scaling
# ══════════════════════════════════════════════════════════════════════════════
def load_episode(episode_id: int) -> pd.DataFrame:
    path = os.path.join(P1_DIR, f"episode_{episode_id}_cleaned.csv")
    df   = pd.read_csv(path)
    return df.sort_values("time_hours").reset_index(drop=True)


class Scaler:
    """Min-max scaler fitted on a single episode."""
    def __init__(self, df: pd.DataFrame):
        self.t_min  = df["time_hours"].min();  self.t_max  = df["time_hours"].max()
        self.s_min  = 0.0;                     self.s_max  = df["glucose"].max()
        self.x_min  = 0.0;                     self.x_max  = df["biomass"].max()
        self.p_min  = 0.0;                     self.p_max  = df["ethanol"].max()

    # ── normalise / denormalise time ────────────────────────────
    def norm_t(self, t):  return (t - self.t_min) / (self.t_max - self.t_min + 1e-8)
    def denorm_t(self, tn): return tn * (self.t_max - self.t_min) + self.t_min

    # ── normalise concentrations (0→1) ─────────────────────────
    def norm_s(self, s):  return s / (self.s_max + 1e-8)
    def norm_x(self, x):  return x / (self.x_max + 1e-8)
    def norm_p(self, p):  return p / (self.p_max + 1e-8)

    # ── denormalise concentrations ─────────────────────────────
    def denorm_s(self, sn): return sn * self.s_max
    def denorm_x(self, xn): return xn * self.x_max
    def denorm_p(self, pn): return pn * self.p_max

    def get_data_tensors(self, df):
        t_n  = torch.tensor(self.norm_t(df["time_hours"].values),
                             dtype=torch.float32).unsqueeze(1)
        s_n  = torch.tensor(self.norm_s(df["glucose"].values),
                             dtype=torch.float32).unsqueeze(1)
        x_n  = torch.tensor(self.norm_x(df["biomass"].values),
                             dtype=torch.float32).unsqueeze(1)
        p_n  = torch.tensor(self.norm_p(df["ethanol"].values),
                             dtype=torch.float32).unsqueeze(1)
        return t_n, s_n, x_n, p_n


# ══════════════════════════════════════════════════════════════════════════════
# Neural-network model
# ══════════════════════════════════════════════════════════════════════════════
class FermentationMLP(nn.Module):
    """MLP: t_norm -> [S_norm, X_norm, P_norm].  Outputs are softplus-clipped ≥ 0."""
    def __init__(self, hidden=32, layers=3):
        super().__init__()
        net = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t_norm):
        out = self.net(t_norm)          # (N, 3)  — can be negative before clipping
        return torch.nn.functional.softplus(out)   # ensures ≥ 0 everywhere


# ══════════════════════════════════════════════════════════════════════════════
# Kinetic parameters  (positive via softplus)
# ══════════════════════════════════════════════════════════════════════════════
class KineticParams(nn.Module):
    def __init__(self):
        super().__init__()
        # raw unconstrained parameters; physical values via softplus
        self._mu_max = nn.Parameter(torch.tensor(self._inv_sp(MU_MAX_INIT)))
        self._Ks     = nn.Parameter(torch.tensor(self._inv_sp(KS_INIT)))
        self._Yxs    = nn.Parameter(torch.tensor(self._inv_sp(YXS_INIT)))
        self._Yps    = nn.Parameter(torch.tensor(self._inv_sp(YPS_INIT)))

    @staticmethod
    def _inv_sp(x):
        """Inverse softplus so that softplus(raw) ≈ x."""
        return float(np.log(np.exp(float(x)) - 1.0 + 1e-8))

    @property
    def mu_max(self): return torch.nn.functional.softplus(self._mu_max)
    @property
    def Ks(self):     return torch.nn.functional.softplus(self._Ks)
    @property
    def Yxs(self):    return torch.nn.functional.softplus(self._Yxs)
    @property
    def Yps(self):    return torch.nn.functional.softplus(self._Yps)


# ══════════════════════════════════════════════════════════════════════════════
# Physics residual loss
# ══════════════════════════════════════════════════════════════════════════════
def physics_loss(model: FermentationMLP,
                 params: KineticParams,
                 scaler: Scaler,
                 t_colloc_norm: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Compute Monod ODE residuals at collocation points.
    All derivatives are taken in normalised-time space and then
    converted to physical-time space before comparing with the ODE RHS.

    dS/dt_phys = dS_norm/dt_norm  * (S_max / T_scale)
    where T_scale = t_max - t_min
    """
    t_colloc_norm.requires_grad_(True)
    pred    = model(t_colloc_norm)      # (N,3)  normalised
    Sn, Xn, Pn = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    # Derivatives wrt normalised time  →  (normalised conc / normalised time)
    ones = torch.ones_like(Sn)
    dSn_dtn = torch.autograd.grad(Sn, t_colloc_norm, grad_outputs=ones,
                                   create_graph=True)[0]
    dXn_dtn = torch.autograd.grad(Xn, t_colloc_norm, grad_outputs=ones,
                                   create_graph=True)[0]
    dPn_dtn = torch.autograd.grad(Pn, t_colloc_norm, grad_outputs=ones,
                                   create_graph=True)[0]

    # Convert to physical units
    T_scale = scaler.t_max - scaler.t_min   # hours
    dS_dt = dSn_dtn * (scaler.s_max / T_scale)
    dX_dt = dXn_dtn * (scaler.x_max / T_scale)
    dP_dt = dPn_dtn * (scaler.p_max / T_scale)

    # Physical concentrations
    S = Sn * scaler.s_max
    X = Xn * scaler.x_max

    # Monod growth rate
    mu = params.mu_max * S / (params.Ks + S + 1e-8)

    # ODE residuals
    res_X = dX_dt - mu * X
    res_S = dS_dt + (1.0 / params.Yxs) * mu * X
    res_P = dP_dt - params.Yps * (1.0 / params.Yxs) * mu * X

    loss_phys = (res_X.pow(2).mean() + res_S.pow(2).mean() + res_P.pow(2).mean()) / 3.0

    residual_rms = {
        "res_X_rms": res_X.detach().abs().mean().item(),
        "res_S_rms": res_S.detach().abs().mean().item(),
        "res_P_rms": res_P.detach().abs().mean().item(),
    }
    return loss_phys, residual_rms


# ══════════════════════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════════════════════
mse = nn.MSELoss()

def data_loss(model, t_obs, s_obs, x_obs, p_obs):
    pred = model(t_obs)
    Sn_pred = pred[:, 0:1]
    Xn_pred = pred[:, 1:2]
    Pn_pred = pred[:, 2:3]
    return mse(Sn_pred, s_obs) + mse(Xn_pred, x_obs) + mse(Pn_pred, p_obs)


def ic_loss(model, scaler, S0, X0, P0):
    """Enforce initial conditions at t=0 (normalised t=0)."""
    t0   = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    pred = model(t0)
    Sn0_pred = pred[0, 0]
    Xn0_pred = pred[0, 1]
    Pn0_pred = pred[0, 2]
    Sn0 = torch.tensor(scaler.norm_s(S0), dtype=torch.float32)
    Xn0 = torch.tensor(scaler.norm_x(X0), dtype=torch.float32)
    Pn0 = torch.tensor(scaler.norm_p(P0), dtype=torch.float32)
    return (Sn0_pred - Sn0).pow(2) + (Xn0_pred - Xn0).pow(2) + (Pn0_pred - Pn0).pow(2)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop  (works for PINN and baseline)
# ══════════════════════════════════════════════════════════════════════════════
def train(model: FermentationMLP,
          scaler: Scaler,
          df_train: pd.DataFrame,
          df_val: pd.DataFrame,
          params,          # KineticParams or None
          use_physics: bool,
          label: str) -> dict:

    # ── Observation tensors ─────────────────────────────────────
    t_obs, s_obs, x_obs, p_obs = scaler.get_data_tensors(df_train)
    t_val, s_val, x_val, p_val = scaler.get_data_tensors(df_val)

    # Initial conditions (from first row)
    S0 = df_train["glucose"].iloc[0]
    X0 = df_train["biomass"].iloc[0]
    P0 = df_train["ethanol"].iloc[0]

    # ── Collocation points ──────────────────────────────────────
    t_domain_phys = np.linspace(scaler.t_min, scaler.t_max, N_COLLOC)
    t_colloc_norm = torch.tensor(
        scaler.norm_t(t_domain_phys), dtype=torch.float32
    ).unsqueeze(1)

    # ── Optimiser ───────────────────────────────────────────────
    if use_physics and params is not None:
        all_params = list(model.parameters()) + list(params.parameters())
    else:
        all_params = list(model.parameters())

    optimizer = torch.optim.Adam(all_params, lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=LR_FACTOR,
                                   patience=LR_PATIENCE, min_lr=MIN_LR)

    history = {"total": [], "data": [], "physics": [], "ic": [], "val": []}
    best_val    = float("inf")
    patience_ct = 0
    best_state  = None

    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"{'='*60}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()

        l_data = data_loss(model, t_obs, s_obs, x_obs, p_obs)
        l_ic   = ic_loss(model, scaler, S0, X0, P0)

        if use_physics and params is not None:
            l_phys, _ = physics_loss(model, params, scaler, t_colloc_norm.clone())
            total = LAMBDA_DATA * l_data + LAMBDA_PHYS * l_phys + LAMBDA_IC * l_ic
        else:
            l_phys = torch.tensor(0.0)
            total  = LAMBDA_DATA * l_data + LAMBDA_IC * l_ic

        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(t_val)
            l_val = (mse(pred_val[:, 0:1], s_val) +
                     mse(pred_val[:, 1:2], x_val) +
                     mse(pred_val[:, 2:3], p_val))

        scheduler.step(l_val.item())

        history["total"].append(total.item())
        history["data"].append(l_data.item())
        history["physics"].append(l_phys.item())
        history["ic"].append(l_ic.item())
        history["val"].append(l_val.item())

        # Early stopping
        if l_val.item() < best_val - 1e-7:
            best_val    = l_val.item()
            patience_ct = 0
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}  (best val={best_val:.6f})")
                break

        if (epoch + 1) % 2000 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:>6d} | total={total.item():.5f} "
                  f"data={l_data.item():.5f} phys={l_phys.item():.5f} "
                  f"ic={l_ic.item():.5f} val={l_val.item():.5f} lr={lr_now:.2e}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  Final best val loss = {best_val:.6f}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════
def predict_physical(model: FermentationMLP, scaler: Scaler,
                     t_phys: np.ndarray) -> np.ndarray:
    """Return predictions in physical units  (N, 3)  [S, X, P]."""
    model.eval()
    t_n = torch.tensor(scaler.norm_t(t_phys), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pred_n = model(t_n).numpy()
    pred = np.column_stack([
        pred_n[:, 0] * scaler.s_max,
        pred_n[:, 1] * scaler.x_max,
        pred_n[:, 2] * scaler.p_max,
    ])
    return pred


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    labels=("S", "X", "P")) -> dict:
    metrics = {}
    for i, lab in enumerate(labels):
        yt = y_true[:, i]; yp = y_pred[:, i]
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae  = float(np.mean(np.abs(yt - yp)))
        r2   = float(_r2_score(yt, yp))
        metrics[lab] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    return metrics


def eval_physics_residuals(model, params, scaler, t_phys):
    """Evaluate Monod residuals at given physical time points."""
    t_n  = torch.tensor(scaler.norm_t(t_phys), dtype=torch.float32).unsqueeze(1)
    _, rms = physics_loss(model, params, scaler, t_n)
    return rms


# ── sklearn imported cleanly (installed via pip) ───────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS_MODEL = {"PINN": "#2196F3", "Baseline NN": "#E91E63"}

def plot_predictions(df_train, df_val, t_dense, pred_pinn, pred_nn,
                     scaler, episode_id, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vars_  = [("glucose", "Glucose S (g/L)", 0),
              ("biomass", "Biomass X (g/L)", 1),
              ("ethanol", "Ethanol P (g/L)", 2)]

    for ax, (col, ylabel, idx) in zip(axes, vars_):
        # Observations
        ax.scatter(df_train["time_hours"], df_train[col],
                   color="black", zorder=5, s=50, label="Train obs")
        ax.scatter(df_val["time_hours"], df_val[col],
                   color="orange", zorder=5, s=50, marker="^", label="Val obs")
        # Model curves
        ax.plot(t_dense, pred_pinn[:, idx], color=COLORS_MODEL["PINN"],
                lw=2, label="PINN")
        ax.plot(t_dense, pred_nn[:, idx], color=COLORS_MODEL["Baseline NN"],
                lw=2, linestyle="--", label="Baseline NN")
        ax.set_xlabel("Time (h)"); ax.set_ylabel(ylabel)
        ax.set_title(f"Episode {episode_id} — {ylabel.split(' ')[0]}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle(f"Episode {episode_id}: PINN vs Baseline NN",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Predictions plot → {out_path}")


def plot_loss_history(history_pinn, history_nn, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, hist, title in zip(axes,
                                [history_pinn, history_nn],
                                ["PINN", "Baseline NN"]):
        epochs = range(1, len(hist["total"]) + 1)
        ax.semilogy(epochs, hist["total"],  label="Total",   color="#333333")
        ax.semilogy(epochs, hist["data"],   label="Data",    color="#2196F3")
        ax.semilogy(epochs, hist["val"],    label="Val",     color="#4CAF50",
                    linestyle="--")
        if title == "PINN":
            ax.semilogy(epochs, hist["physics"], label="Physics", color="#E91E63",
                        linestyle=":")
            ax.semilogy(epochs, hist["ic"],      label="IC",      color="#FF9800",
                        linestyle="-.")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle("Training Loss History", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Loss history plot → {out_path}")


def plot_residuals(t_phys, params_pinn, model_pinn, scaler, out_path):
    """Plot physics residuals vs time for the PINN."""
    # Fresh forward pass with a brand-new tensor so we own the graph
    t_n = torch.tensor(scaler.norm_t(t_phys), dtype=torch.float32,
                       requires_grad=True).unsqueeze(1)
    pred = model_pinn(t_n)
    Sn, Xn, Pn = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    ones = torch.ones_like(Sn)
    # Use retain_graph so we can call grad three times on the same graph
    dSn = torch.autograd.grad(Sn, t_n, grad_outputs=ones,
                               create_graph=False, retain_graph=True)[0]
    dXn = torch.autograd.grad(Xn, t_n, grad_outputs=torch.ones_like(Xn),
                               create_graph=False, retain_graph=True)[0]
    dPn = torch.autograd.grad(Pn, t_n, grad_outputs=torch.ones_like(Pn),
                               create_graph=False, retain_graph=False)[0]
    T   = scaler.t_max - scaler.t_min
    dS  = dSn.detach().numpy()[:, 0] * scaler.s_max / T
    dX  = dXn.detach().numpy()[:, 0] * scaler.x_max / T
    dP  = dPn.detach().numpy()[:, 0] * scaler.p_max / T
    S   = Sn.detach().numpy()[:, 0] * scaler.s_max
    X   = Xn.detach().numpy()[:, 0] * scaler.x_max
    mu_max = params_pinn.mu_max.item()
    Ks     = params_pinn.Ks.item()
    Yxs    = params_pinn.Yxs.item()
    Yps    = params_pinn.Yps.item()
    mu  = mu_max * S / (Ks + S + 1e-8)
    rX  = dX - mu * X
    rS  = dS + (1.0 / Yxs) * mu * X
    rP  = dP - Yps * (1.0 / Yxs) * mu * X

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, res, label, color in zip(
            axes,
            [rX, rS, rP],
            ["dX/dt residual (g/L/h)", "dS/dt residual (g/L/h)", "dP/dt residual (g/L/h)"],
            ["#2196F3", "#E91E63", "#4CAF50"]):
        ax.plot(t_phys, res, color=color, lw=1.5)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel("Time (h)"); ax.set_ylabel(label)
        ax.set_title(label.split(" ")[0] + " ODE Residual")
        ax.grid(alpha=0.3)
    plt.suptitle("PINN: Monod ODE Residuals (should be ≈ 0)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Residuals plot → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Phase 2: Physics-Informed Neural Network")
    print("=" * 60)

    # ── Load Episode 20 ─────────────────────────────────────────
    df20  = load_episode(20)
    print(f"\nEpisode 20: {len(df20)} observations, "
          f"t=[{df20.time_hours.min():.1f}, {df20.time_hours.max():.1f}] h")

    # Train/val split: last 2 points held out for validation
    df_train = df20.iloc[:-2].reset_index(drop=True)
    df_val   = df20.iloc[-2:].reset_index(drop=True)
    print(f"  Train: {len(df_train)} obs  |  Val (held-out): {len(df_val)} obs")

    scaler = Scaler(df20)
    print(f"  Concentration scales → S_max={scaler.s_max:.3f}  "
          f"X_max={scaler.x_max:.3f}  P_max={scaler.p_max:.3f}")

    # ── Build models ────────────────────────────────────────────
    model_pinn = FermentationMLP(hidden=32, layers=3).to(DEVICE)
    params_kin = KineticParams().to(DEVICE)
    print(f"\nInitial kinetic params → μmax={params_kin.mu_max.item():.4f}  "
          f"Ks={params_kin.Ks.item():.4f}  "
          f"Yxs={params_kin.Yxs.item():.4f}  "
          f"Yps={params_kin.Yps.item():.4f}")

    model_base = FermentationMLP(hidden=32, layers=3).to(DEVICE)

    # ── Train PINN ──────────────────────────────────────────────
    history_pinn = train(
        model_pinn, scaler, df_train, df_val,
        params=params_kin, use_physics=True, label="PINN"
    )

    # ── Train Baseline NN ───────────────────────────────────────
    history_nn = train(
        model_base, scaler, df_train, df_val,
        params=None, use_physics=False, label="Baseline NN"
    )

    # ── Dense time grid for plotting ────────────────────────────
    t_dense = np.linspace(scaler.t_min, scaler.t_max, 300)
    pred_pinn = predict_physical(model_pinn, scaler, t_dense)
    pred_nn   = predict_physical(model_base, scaler, t_dense)

    # ── Predictions at validation points ───────────────────────
    t_val_phys  = df_val["time_hours"].values
    y_val_true  = df_val[["glucose", "biomass", "ethanol"]].values
    y_val_pinn  = predict_physical(model_pinn, scaler, t_val_phys)
    y_val_nn    = predict_physical(model_base, scaler, t_val_phys)

    # ── Predictions at training points (in-sample) ─────────────
    t_train_phys = df_train["time_hours"].values
    y_train_true = df_train[["glucose", "biomass", "ethanol"]].values
    y_train_pinn = predict_physical(model_pinn, scaler, t_train_phys)
    y_train_nn   = predict_physical(model_base, scaler, t_train_phys)

    # ── Metrics ─────────────────────────────────────────────────
    metrics_pinn_val  = compute_metrics(y_val_true, y_val_pinn)
    metrics_nn_val    = compute_metrics(y_val_true, y_val_nn)
    metrics_pinn_trn  = compute_metrics(y_train_true, y_train_pinn)
    metrics_nn_trn    = compute_metrics(y_train_true, y_train_nn)

    # ── Physics residuals (PINN only) ──────────────────────────
    t_resid = np.linspace(scaler.t_min, scaler.t_max, 150)
    t_n_r   = torch.tensor(scaler.norm_t(t_resid), dtype=torch.float32).unsqueeze(1)
    _, resid_rms = physics_loss(model_pinn, params_kin, scaler, t_n_r)

    # ── Fitted kinetic parameters ──────────────────────────────
    fitted_params = {
        "mu_max": params_kin.mu_max.item(),
        "Ks":     params_kin.Ks.item(),
        "Yxs":    params_kin.Yxs.item(),
        "Yps":    params_kin.Yps.item(),
    }

    # ══════════════════════════════════════════════════════════
    # Print summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FITTED KINETIC PARAMETERS (PINN)")
    print("=" * 60)
    for k, v in fitted_params.items():
        print(f"  {k:<8s} = {v:.5f}")

    print("\n" + "=" * 60)
    print("EVALUATION — Held-out Validation Points (Episode 20, last 2)")
    print("=" * 60)
    print(f"{'Variable':<10} {'Model':<14} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 50)
    for var in ("S", "X", "P"):
        for label, met in [("PINN", metrics_pinn_val), ("Baseline", metrics_nn_val)]:
            m = met[var]
            print(f"{var:<10} {label:<14} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
                  f"{m['R2']:>8.4f}")
        print()

    print("=" * 60)
    print("EVALUATION — Training Points (in-sample)")
    print("=" * 60)
    print(f"{'Variable':<10} {'Model':<14} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 50)
    for var in ("S", "X", "P"):
        for label, met in [("PINN", metrics_pinn_trn), ("Baseline", metrics_nn_trn)]:
            m = met[var]
            print(f"{var:<10} {label:<14} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
                  f"{m['R2']:>8.4f}")
        print()

    print("=" * 60)
    print("PHYSICS RESIDUALS (PINN, RMS over 150 collocation pts)")
    print("=" * 60)
    for k, v in resid_rms.items():
        print(f"  {k:<20s} = {v:.6f} g/L/h")

    # ══════════════════════════════════════════════════════════
    # Save artefacts
    # ══════════════════════════════════════════════════════════
    torch.save(model_pinn.state_dict(),
               os.path.join(OUT_DIR, "pinn_source_weights.pt"))
    torch.save(model_base.state_dict(),
               os.path.join(OUT_DIR, "baseline_nn_weights.pt"))
    torch.save(params_kin.state_dict(),
               os.path.join(OUT_DIR, "pinn_kinetic_params.pt"))
    print("\n✓ Model weights saved.")

    with open(os.path.join(OUT_DIR, "fitted_kinetic_params.json"), "w") as f:
        json.dump(fitted_params, f, indent=2)

    # Loss history CSV
    hist_df_pinn = pd.DataFrame(history_pinn)
    hist_df_pinn.index.name = "epoch"
    hist_df_pinn.to_csv(os.path.join(OUT_DIR, "pinn_loss_history.csv"))

    hist_df_nn = pd.DataFrame(history_nn)
    hist_df_nn.index.name = "epoch"
    hist_df_nn.to_csv(os.path.join(OUT_DIR, "baseline_loss_history.csv"))
    print("✓ Loss history CSVs saved.")

    # Metrics JSON
    all_metrics = {
        "pinn_val":      metrics_pinn_val,
        "baseline_val":  metrics_nn_val,
        "pinn_train":    metrics_pinn_trn,
        "baseline_train":metrics_nn_trn,
        "physics_residuals_rms": resid_rms,
    }
    with open(os.path.join(OUT_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("✓ Evaluation metrics saved.")

    # Plots
    plot_predictions(df_train, df_val, t_dense, pred_pinn, pred_nn,
                     scaler, 20,
                     os.path.join(OUT_DIR, "predictions_ep20.png"))
    plot_loss_history(history_pinn, history_nn,
                      os.path.join(OUT_DIR, "loss_history.png"))
    plot_residuals(t_dense, params_kin, model_pinn, scaler,
                   os.path.join(OUT_DIR, "physics_residuals.png"))

    print("\n" + "=" * 60)
    print("Phase 2 complete.  All outputs in:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
