"""
Cross-Episode Validation (corrected) -- extend Phase 3C hard-IC PINN.

Scaler fix: fit normalization on the full valid episode range (not just training
points), matching Phase 3C's approach. This avoids test-time glucose values
falling outside the scaler's range.

Target A: Episode 31  (S0 = 15.41 g/L, low  glucose, t_valid = 0..6)
Target B: Episode 61  (S0 = 23.43 g/L, high glucose, t_valid = 0..9)
Reference: Episode 75 (S0 = 23.46 g/L, Phase 3C n=5 results reused)
"""

import os, json, copy, sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS  = [42, 7, 123]
DEVICE = torch.device('cpu')

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, 'data', 'complete_fermentation_dataset.csv')
SRC_W = os.path.join(BASE, 'outputs', 'phase3c', 'src_multi_net_seed42.pt')
OUT   = os.path.join(BASE, 'outputs', 'cross_episode_validation')
os.makedirs(OUT, exist_ok=True)

# ── Safeguard: Check if Phase 3C weights exist ────────────────────────────────
if not os.path.exists(SRC_W):
    print("=" * 65)
    print("❌ ERROR: Pre-trained weights not found!")
    print(f"Missing file: {SRC_W}")
    print("Please execute 'Phase 3C: Hard-IC PINN' first to generate these weights.")
    print("=" * 65)
    sys.exit(1)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR_SCRATCH   = 1e-3
LR_TRANSFER  = 1e-3   # same as scratch; pretrained init is the advantage, not a lower LR
N_EPOCHS     = 5000
N_COLLOC     = 150
LAM_PHYS     = 1e-2
PATIENCE     = 400
CLIP         = 1.0

# ── Episode config ────────────────────────────────────────────────────────────
# t_valid_max: last valid time point in glucose-consumption window
EP_CFG = {31: {'t_valid_max': 6, 's0': 15.41}, 61: {'t_valid_max': 9, 's0': 23.43}}

# ── Data ──────────────────────────────────────────────────────────────────────
def load_episode(ep_id, t_max):
    df = pd.read_csv(DATA)
    sub = df[df['episode_id'] == ep_id].sort_values('time_hours').reset_index(drop=True)
    sub = sub[sub['time_hours'] <= t_max]
    return (sub['time_hours'].values.astype(np.float32),
            sub['glucose'].values.astype(np.float32),
            sub['biomass'].values.astype(np.float32),
            sub['ethanol'].values.astype(np.float32))

def select_split(t_all, n_obs=5, n_test=3):
    """Fixed split: last n_test = test; select n_obs from earlier pool."""
    n_total = len(t_all)
    test_idx = list(range(n_total - n_test, n_total))
    pool_idx = list(range(0, n_total - n_test))
    # always include t=0 (index 0); pick remaining evenly
    if n_obs <= 1:
        train_idx = [0]
    else:
        rest = pool_idx[1:]
        n_more = n_obs - 1
        if len(rest) <= n_more:
            chosen = rest
        else:
            idxs = np.round(np.linspace(0, len(rest)-1, n_more)).astype(int)
            chosen = [rest[i] for i in idxs]
        train_idx = sorted(set([0] + chosen))
    return train_idx, test_idx

# ── Scaler: fit on FULL valid range (matches Phase 3C) ───────────────────────
def make_scaler(t_all, S_all, X_all, P_all):
    return {
        't_max': float(t_all.max()) + 1e-8,
        'S_max': float(S_all.max()),
        'X_max': float(X_all.max()),
        'P_max': float(P_all.max()),
    }

# ── Model ─────────────────────────────────────────────────────────────────────
class HardICNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(4, 32), nn.Tanh()]
        for _ in range(2):
            layers += [nn.Linear(32, 32), nn.Tanh()]
        layers += [nn.Linear(32, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm, ic_norm):
        """t_norm:(B,1), ic_norm:(B,3) -> (B,3) norm predictions"""
        x = torch.cat([t_norm, ic_norm], dim=-1)
        g = self.net(x)
        exp_arg = torch.clamp(t_norm * g, -30.0, 30.0)
        return ic_norm * torch.exp(exp_arg)

class KP(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_mu_max = nn.Parameter(torch.tensor(0.0))
        self.log_Ks     = nn.Parameter(torch.tensor(0.0))
        self.log_Yxs    = nn.Parameter(torch.tensor(0.0))
        self.log_Yps    = nn.Parameter(torch.tensor(0.0))

    def get(self):
        return (torch.exp(self.log_mu_max), torch.exp(self.log_Ks),
                torch.exp(self.log_Yxs), torch.exp(self.log_Yps))

# ── Loss ──────────────────────────────────────────────────────────────────────
def phys_loss(net, kp, t_c, ic_c, sc):
    t_in = t_c.requires_grad_(True)
    pred = net(t_in, ic_c)
    S = pred[:,0:1] * sc['S_max']
    X = pred[:,1:2] * sc['X_max']
    P = pred[:,2:3] * sc['P_max']
    dS = torch.autograd.grad(S.sum(), t_in, create_graph=True)[0] / sc['t_max']
    dX = torch.autograd.grad(X.sum(), t_in, create_graph=True)[0] / sc['t_max']
    dP = torch.autograd.grad(P.sum(), t_in, create_graph=True)[0] / sc['t_max']
    mu_max, Ks, Yxs, Yps = kp.get()
    mu = mu_max * S / (Ks + S + 1e-8)
    rX = dX - mu * X
    rS = dS + mu * X / (Yxs + 1e-8)
    rP = dP + Yps * dS
    return (rX**2 + rS**2 + rP**2).mean()

# ── Train ─────────────────────────────────────────────────────────────────────
def train_model(net, kp, t_tr_n, S_tr_n, X_tr_n, P_tr_n,
                ic_norm, t_coll_n, ic_coll, sc, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(list(net.parameters()) + list(kp.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.5, min_lr=1e-6)

    t_t  = torch.tensor(t_tr_n,  dtype=torch.float32).unsqueeze(-1)
    St   = torch.tensor(S_tr_n,  dtype=torch.float32)
    Xt   = torch.tensor(X_tr_n,  dtype=torch.float32)
    Pt   = torch.tensor(P_tr_n,  dtype=torch.float32)
    n_tr = len(t_t)
    ic_t = torch.tensor(ic_norm, dtype=torch.float32).unsqueeze(0).expand(n_tr, -1)
    t_c  = torch.tensor(t_coll_n, dtype=torch.float32).unsqueeze(-1)
    ic_c = torch.tensor(ic_coll,  dtype=torch.float32)

    best_loss = float('inf')
    best_net  = copy.deepcopy(net.state_dict())
    best_kp   = copy.deepcopy(kp.state_dict())
    pat = 0

    for epoch in range(N_EPOCHS):
        net.train(); kp.train()
        opt.zero_grad()
        pred  = net(t_t, ic_t)
        d_l   = ((pred[:,0]-St)**2 + (pred[:,1]-Xt)**2 + (pred[:,2]-Pt)**2).mean()
        p_l   = phys_loss(net, kp, t_c, ic_c, sc)
        loss  = d_l + LAM_PHYS * p_l
        loss.backward()
        nn.utils.clip_grad_norm_(list(net.parameters())+list(kp.parameters()), CLIP)
        opt.step()
        sched.step(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_net  = copy.deepcopy(net.state_dict())
            best_kp   = copy.deepcopy(kp.state_dict())
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE: break

    net.load_state_dict(best_net)
    kp.load_state_dict(best_kp)
    return net, kp

# ── Evaluate ──────────────────────────────────────────────────────────────────
def predict(net, t_eval, ic_norm, sc):
    net.eval()
    t_n  = torch.tensor(t_eval / sc['t_max'], dtype=torch.float32).unsqueeze(-1)
    ic_e = torch.tensor(ic_norm, dtype=torch.float32).unsqueeze(0).expand(len(t_n), -1)
    with torch.no_grad():
        pred = net(t_n, ic_e)
    return (pred[:,0].numpy() * sc['S_max'],
            pred[:,1].numpy() * sc['X_max'],
            pred[:,2].numpy() * sc['P_max'])

def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))

# ── Run one episode ───────────────────────────────────────────────────────────
def run_episode(ep_id, t_max, n_obs=5, n_test=3):
    t_all, S_all, X_all, P_all = load_episode(ep_id, t_max)
    train_idx, test_idx = select_split(t_all, n_obs=n_obs, n_test=n_test)

    t_tr = t_all[train_idx]; S_tr = S_all[train_idx]
    X_tr = X_all[train_idx]; P_tr = P_all[train_idx]
    t_te = t_all[test_idx];  S_te = S_all[test_idx]
    X_te = X_all[test_idx];  P_te = P_all[test_idx]

    print(f"  Ep{ep_id}: train_t={list(t_tr.astype(int))}  test_t={list(t_te.astype(int))}")

    # Scaler on full valid range
    sc = make_scaler(t_all, S_all, X_all, P_all)
    t_tr_n = t_tr / sc['t_max']
    S_tr_n = S_tr / sc['S_max']; X_tr_n = X_tr / sc['X_max']; P_tr_n = P_tr / sc['P_max']

    ic_norm = np.array([S_tr[0]/sc['S_max'], X_tr[0]/sc['X_max'], P_tr[0]/sc['P_max']],
                       dtype=np.float32)

    # Collocation across full prediction interval
    t_coll   = np.linspace(0, t_max, N_COLLOC).astype(np.float32)
    t_coll_n = t_coll / sc['t_max']
    ic_coll  = np.tile(ic_norm, (N_COLLOC, 1)).astype(np.float32)

    res = {'scratch': [], 'transfer': []}

    for seed in SEEDS:
        # --- Scratch ---
        torch.manual_seed(seed)
        net_sc = HardICNet(); kp_sc = KP()
        net_sc, kp_sc = train_model(net_sc, kp_sc,
                                    t_tr_n, S_tr_n, X_tr_n, P_tr_n,
                                    ic_norm, t_coll_n, ic_coll, sc,
                                    lr=LR_SCRATCH, seed=seed)
        S_p, X_p, P_p = predict(net_sc, t_te, ic_norm, sc)
        # IC error: predict at t=0
        S0_p, _, _ = predict(net_sc, np.array([0.0], dtype=np.float32), ic_norm, sc)
        ic_err = abs(S0_p[0] - S_tr[0])
        res['scratch'].append({
            'S_rmse': rmse(S_p, S_te), 'X_rmse': rmse(X_p, X_te),
            'P_rmse': rmse(P_p, P_te), 'S_mae': mae(S_p, S_te),
            'X_mae': mae(X_p, X_te), 'P_mae': mae(P_p, P_te),
            'ic_err': ic_err, 'pred_S': S_p, 'pred_X': X_p, 'pred_P': P_p
        })

        # --- Transfer ---
        torch.manual_seed(seed)
        net_tr = HardICNet(); kp_tr = KP()
        net_tr.load_state_dict(
            torch.load(SRC_W, map_location=DEVICE, weights_only=True))
        net_tr, kp_tr = train_model(net_tr, kp_tr,
                                    t_tr_n, S_tr_n, X_tr_n, P_tr_n,
                                    ic_norm, t_coll_n, ic_coll, sc,
                                    lr=LR_TRANSFER, seed=seed)
        S_p, X_p, P_p = predict(net_tr, t_te, ic_norm, sc)
        S0_p, _, _ = predict(net_tr, np.array([0.0], dtype=np.float32), ic_norm, sc)
        ic_err = abs(S0_p[0] - S_tr[0])
        res['transfer'].append({
            'S_rmse': rmse(S_p, S_te), 'X_rmse': rmse(X_p, X_te),
            'P_rmse': rmse(P_p, P_te), 'S_mae': mae(S_p, S_te),
            'X_mae': mae(X_p, X_te), 'P_mae': mae(P_p, P_te),
            'ic_err': ic_err, 'pred_S': S_p, 'pred_X': X_p, 'pred_P': P_p
        })
        print(f"    seed={seed}: scratch S={res['scratch'][-1]['S_rmse']:.3f} "
              f"transfer S={res['transfer'][-1]['S_rmse']:.3f}")

    def agg(lst, key):
        v = [r[key] for r in lst]
        return float(np.mean(v)), float(np.std(v))

    agg_res = {
        cond: {m: agg(lst, m)
               for m in ['S_rmse','X_rmse','P_rmse','S_mae','X_mae','P_mae','ic_err']}
        for cond, lst in res.items()
    }

    def best_pred(lst, key='S_rmse'):
        mean_v = np.mean([r[key] for r in lst])
        idx = int(np.argmin([abs(r[key] - mean_v) for r in lst]))
        return lst[idx]

    return agg_res, best_pred(res['scratch']), best_pred(res['transfer']), \
           t_te, S_te, X_te, P_te, t_all, S_all, X_all, P_all, sc

# ── Ep75 reference (Phase 3C, n=5) ───────────────────────────────────────────
EP75 = {
    'scratch':  {'S_rmse': (2.442, 0.162), 'X_rmse': (0.651, 0.160), 'P_rmse': (0.944, 0.845),
                 'ic_err': (3.04e-7, 0.0)},
    'transfer': {'S_rmse': (1.361, 0.537), 'X_rmse': (1.189, 0.241), 'P_rmse': (3.480, 1.070),
                 'ic_err': (3.04e-7, 0.0)},
}

# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Cross-Episode Validation  (Phase 3C hard-IC PINN extension)")
print("Target A: Ep31  S0=15.41 g/L  (low glucose)")
print("Target B: Ep61  S0=23.43 g/L  (high glucose)")
print("=" * 65)

ep_res  = {}
ep_pred = {}
ep_dat  = {}
ep_sc   = {}

for ep_id, cfg in EP_CFG.items():
    print(f"\nEpisode {ep_id} ...")
    agg, sc_p, tr_p, t_te, S_te, X_te, P_te, t_all, S_all, X_all, P_all, sc = \
        run_episode(ep_id, cfg['t_valid_max'])
    ep_res[ep_id]  = agg
    ep_pred[ep_id] = {'scratch': sc_p, 'transfer': tr_p}
    ep_dat[ep_id]  = {'t_te': t_te, 'S_te': S_te, 'X_te': X_te, 'P_te': P_te,
                      't_all': t_all, 'S_all': S_all}
    ep_sc[ep_id]   = sc

def pct(sc_v, tr_v):
    return 100.0 * (sc_v - tr_v) / (sc_v + 1e-12)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS  (n=5 obs, fixed test = last 3 valid time points)")
print("=" * 65)

EP_S0 = {75: 23.46, 31: 15.41, 61: 23.43}

def print_ep(ep_id, sc_res, tr_res, label=''):
    p_S = pct(sc_res['S_rmse'][0], tr_res['S_rmse'][0])
    p_X = pct(sc_res['X_rmse'][0], tr_res['X_rmse'][0])
    p_P = pct(sc_res['P_rmse'][0], tr_res['P_rmse'][0])
    print(f"\nEp{ep_id} (S0={EP_S0[ep_id]:.2f} g/L)  {label}")
    print(f"  Glucose : Sc={sc_res['S_rmse'][0]:.3f}±{sc_res['S_rmse'][1]:.3f}  "
          f"Tr={tr_res['S_rmse'][0]:.3f}±{tr_res['S_rmse'][1]:.3f}  pct={p_S:+.1f}%")
    print(f"  Biomass : Sc={sc_res['X_rmse'][0]:.3f}±{sc_res['X_rmse'][1]:.3f}  "
          f"Tr={tr_res['X_rmse'][0]:.3f}±{tr_res['X_rmse'][1]:.3f}  pct={p_X:+.1f}%")
    print(f"  Ethanol : Sc={sc_res['P_rmse'][0]:.3f}±{sc_res['P_rmse'][1]:.3f}  "
          f"Tr={tr_res['P_rmse'][0]:.3f}±{tr_res['P_rmse'][1]:.3f}  pct={p_P:+.1f}%")

print_ep(75, EP75['scratch'], EP75['transfer'], '[Phase 3C reference]')
print_ep(31, ep_res[31]['scratch'], ep_res[31]['transfer'], '[Target A, new]')
print_ep(61, ep_res[61]['scratch'], ep_res[61]['transfer'], '[Target B, new]')

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_rows = []
for ep_id, s0 in EP_S0.items():
    if ep_id == 75:
        sc_r, tr_r = EP75['scratch'], EP75['transfer']
        note = 'Phase 3C reference'
    else:
        sc_r, tr_r = ep_res[ep_id]['scratch'], ep_res[ep_id]['transfer']
        note = f'New target {"A" if ep_id==31 else "B"}'
    p_S = pct(sc_r['S_rmse'][0], tr_r['S_rmse'][0])
    csv_rows.append({
        'episode': ep_id, 'S0_gL': s0,
        'scratch_S_RMSE_mean': round(sc_r['S_rmse'][0],3),
        'scratch_S_RMSE_std':  round(sc_r['S_rmse'][1],3),
        'transfer_S_RMSE_mean': round(tr_r['S_rmse'][0],3),
        'transfer_S_RMSE_std':  round(tr_r['S_rmse'][1],3),
        'glucose_pct_improvement': round(p_S,1),
        'scratch_X_RMSE_mean': round(sc_r['X_rmse'][0],3),
        'transfer_X_RMSE_mean': round(tr_r['X_rmse'][0],3),
        'scratch_P_RMSE_mean': round(sc_r['P_rmse'][0],3),
        'transfer_P_RMSE_mean': round(tr_r['P_rmse'][0],3),
        'note': note,
    })
pd.DataFrame(csv_rows).to_csv(os.path.join(OUT, 'comparison_results.csv'), index=False)
print(f"\nSaved CSV: {os.path.join(OUT, 'comparison_results.csv')}")

# ── Figure 1: Glucose RMSE comparison ────────────────────────────────────────
labels = ['Ep75\n(ref)\nS0=23.46', 'Ep31 (A)\nS0=15.41', 'Ep61 (B)\nS0=23.43']
sc_means = [EP75['scratch']['S_rmse'][0],
            ep_res[31]['scratch']['S_rmse'][0],
            ep_res[61]['scratch']['S_rmse'][0]]
sc_stds  = [EP75['scratch']['S_rmse'][1],
            ep_res[31]['scratch']['S_rmse'][1],
            ep_res[61]['scratch']['S_rmse'][1]]
tr_means = [EP75['transfer']['S_rmse'][0],
            ep_res[31]['transfer']['S_rmse'][0],
            ep_res[61]['transfer']['S_rmse'][0]]
tr_stds  = [EP75['transfer']['S_rmse'][1],
            ep_res[31]['transfer']['S_rmse'][1],
            ep_res[61]['transfer']['S_rmse'][1]]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3); w = 0.32
ax.bar(x - w/2, sc_means, w, yerr=sc_stds, capsize=5, color='#4C72B0',
       label='Hard-IC Scratch PINN', alpha=0.9)
ax.bar(x + w/2, tr_means, w, yerr=tr_stds, capsize=5, color='#DD8452',
       label='Hard-IC Multi-Source Transfer', alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Glucose RMSE (g/L)')
ax.set_title('Glucose RMSE: Scratch vs Multi-Source Transfer\n'
             '(n=5 observations, fixed test = last 3 valid time points)')
ax.legend()
ax.set_ylim(0, max(sc_means + tr_means) * 1.5)
for i in range(3):
    p = pct(sc_means[i], tr_means[i])
    ypos = max(sc_means[i]+sc_stds[i], tr_means[i]+tr_stds[i]) + 0.05
    ax.text(i, ypos, f'{p:+.0f}%', ha='center', va='bottom', fontsize=9,
            color='green' if p > 0 else 'red', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'glucose_rmse_comparison.png'), dpi=120)
plt.close()
print(f"Saved: {os.path.join(OUT, 'glucose_rmse_comparison.png')}")

# ── Figure 2: Glucose predictions ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ep_id in zip(axes, [31, 61]):
    d    = ep_dat[ep_id]
    t_te = d['t_te']
    t_all= d['t_all']; S_all = d['S_all']
    t_max = EP_CFG[ep_id]['t_valid_max']
    sc_p = ep_pred[ep_id]['scratch']
    tr_p = ep_pred[ep_id]['transfer']

    ax.plot(t_all, S_all, 'k-o', lw=1.2, ms=4, label='All measurements', alpha=0.5)
    ax.scatter(d['t_te'], d['S_te'], color='black', marker='D', s=70, zorder=6,
               label='Held-out test points')
    ax.scatter(t_te, sc_p['pred_S'], color='#4C72B0', marker='s', s=70, zorder=5,
               label=f"Scratch  RMSE={ep_res[ep_id]['scratch']['S_rmse'][0]:.2f} g/L")
    ax.scatter(t_te, tr_p['pred_S'], color='#DD8452', marker='^', s=70, zorder=5,
               label=f"Transfer RMSE={ep_res[ep_id]['transfer']['S_rmse'][0]:.2f} g/L")
    t_cut = t_te[0]
    ax.axvspan(t_cut, t_max, alpha=0.07, color='gray')
    ax.axvline(t_cut, color='gray', lw=1, ls='--')
    ax.text(t_cut + 0.1, S_all.max()*0.9, 'Test\nregion', fontsize=8, color='gray')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Glucose S (g/L)')
    ax.set_title(f'Episode {ep_id}  (S0 = {EP_CFG[ep_id]["s0"]:.2f} g/L)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle('Glucose Predictions: Scratch vs Multi-Source Transfer\n'
             '(New target episodes, n=5 observations)', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'glucose_predictions.png'), dpi=120)
plt.close()
print(f"Saved: {os.path.join(OUT, 'glucose_predictions.png')}")

# ── Comparison report ─────────────────────────────────────────────────────────
lines = []
lines.append("Cross-Episode Validation Report")
lines.append("=" * 65)
lines.append("")
lines.append("Episode selection:")
lines.append("  Ep31 (Target A): S0=15.41 g/L  valid window t=0..6h  (low glucose)")
lines.append("     Rationale: clean, never used for training/selection, low S0")
lines.append("  Ep61 (Target B): S0=23.43 g/L  valid window t=0..9h  (high glucose)")
lines.append("     Rationale: clean, never used for training/selection, high S0")
lines.append("  Excluded: Ep20(source), Ep75(original target),")
lines.append("            Ep27,55,11,47,10,80,78,82 (multi-source pretrain), Ep31 used.")
lines.append("")
lines.append("Protocol (matches Phase 3C):")
lines.append("  n=5 (including t=0), test = last 3 valid time points")
lines.append("  Seeds [42, 7, 123], mean ± std reported")
lines.append("  Scaler: fitted on full valid episode range (not just training points)")
lines.append("  Multi-source weights: outputs/phase3c/src_multi_net_seed42.pt")
lines.append("  Fine-tuning LR: 1e-3 (same as scratch; pretrained init is the advantage)")
lines.append("")
lines.append("Results:")
lines.append("")

for ep_id in [75, 31, 61]:
    if ep_id == 75:
        sc_r, tr_r = EP75['scratch'], EP75['transfer']
        note = '[Phase 3C reference]'
    else:
        sc_r, tr_r = ep_res[ep_id]['scratch'], ep_res[ep_id]['transfer']
        note = f'[New target {"A" if ep_id==31 else "B"}]'
    p_S = pct(sc_r['S_rmse'][0], tr_r['S_rmse'][0])
    p_X = pct(sc_r['X_rmse'][0], tr_r['X_rmse'][0])
    p_P = pct(sc_r['P_rmse'][0], tr_r['P_rmse'][0])
    lines.append(f"Ep{ep_id} (S0={EP_S0[ep_id]:.2f} g/L)  {note}")
    lines.append(f"  Glucose S: Scratch={sc_r['S_rmse'][0]:.3f}±{sc_r['S_rmse'][1]:.3f}  "
                 f"Transfer={tr_r['S_rmse'][0]:.3f}±{tr_r['S_rmse'][1]:.3f}  "
                 f"=> {p_S:+.1f}% ({'IMPROVED' if p_S>0 else 'DEGRADED'})")
    lines.append(f"  Biomass X: Scratch={sc_r['X_rmse'][0]:.3f}±{sc_r['X_rmse'][1]:.3f}  "
                 f"Transfer={tr_r['X_rmse'][0]:.3f}±{tr_r['X_rmse'][1]:.3f}  "
                 f"=> {p_X:+.1f}% ({'IMPROVED' if p_X>0 else 'DEGRADED'})")
    lines.append(f"  Ethanol P: Scratch={sc_r['P_rmse'][0]:.3f}±{sc_r['P_rmse'][1]:.3f}  "
                 f"Transfer={tr_r['P_rmse'][0]:.3f}±{tr_r['P_rmse'][1]:.3f}  "
                 f"=> {p_P:+.1f}% ({'IMPROVED' if p_P>0 else 'DEGRADED'})")
    lines.append("")

# Overall consistency
all_pcts_S = [pct(EP75['scratch']['S_rmse'][0], EP75['transfer']['S_rmse'][0])]
for ep_id in [31, 61]:
    all_pcts_S.append(pct(ep_res[ep_id]['scratch']['S_rmse'][0],
                          ep_res[ep_id]['transfer']['S_rmse'][0]))
consistent_S = all(p > 0 for p in all_pcts_S)
lines.append(f"Transfer improved glucose on all 3 episodes: {consistent_S}")
lines.append(f"  Glucose improvements: {[f'{p:+.1f}%' for p in all_pcts_S]}")
lines.append("")
lines.append("Limitations:")
lines.append("  1. Each episode has only 3 fixed test points; individual RMSE estimates")
lines.append("     have high statistical uncertainty.")
lines.append("  2. Ep31 valid window is only 7 points (t=0..6), leaving only 4 pool")
lines.append("     points after reserving test; the n=5 training subset is constrained.")
lines.append("  3. Multi-source pretrain used seed=42 only; source-seed variability")
lines.append("     was not re-assessed for the new episodes.")
lines.append("  4. Ethanol and biomass results are mixed across episodes; transfer does")
lines.append("     not universally outperform scratch for all variables.")
lines.append("  5. Ep61 (S0=23.43 g/L) is at the edge of the multi-source pretraining")
lines.append("     distribution (source episodes max ~23.4 g/L), possibly limiting")
lines.append("     the benefit of pretraining for high-glucose targets.")

with open(os.path.join(OUT, 'comparison_report.txt'), 'w') as f:
    f.write('\n'.join(lines))
print(f"Saved: {os.path.join(OUT, 'comparison_report.txt')}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_out = {}
for ep_id in [31, 61]:
    r = ep_res[ep_id]
    json_out[f'ep{ep_id}'] = {
        cond: {m: {'mean': v[0], 'std': v[1]} for m, v in r[cond].items()}
        for cond in r
    }
json_out['ep75_ref'] = {
    cond: {m: {'mean': v[0], 'std': v[1]} for m, v in EP75[cond].items()}
    for cond in EP75
}
with open(os.path.join(OUT, 'full_results.json'), 'w') as f:
    json.dump(json_out, f, indent=2)
print(f"Saved: {os.path.join(OUT, 'full_results.json')}")

print("\n" + "=" * 65)
print("Cross-episode validation complete.")
print(f"Outputs: {OUT}")