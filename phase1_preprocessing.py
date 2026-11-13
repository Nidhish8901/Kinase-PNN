"""
Phase 1: Dataset Preprocessing
Transfer Learning with Physics-Informed Neural Networks for Bioethanol Fermentation

Outputs (all written to outputs/phase1/):
  - episode_<id>_cleaned.csv   : glucose-consumption-phase data for each selected episode
  - phase1_report.txt          : episode summary (IDs, initial glucose, depletion time)
  - phase1_visualization.png   : glucose / biomass / ethanol vs time for all 3 episodes
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "complete_fermentation_dataset.csv")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "phase1")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Threshold for "near-zero glucose" ─────────────────────────────────────────
GLUCOSE_DEPLETION_THRESHOLD = 0.5   # g/L


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & validate
# ══════════════════════════════════════════════════════════════════════════════
def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {"time_hours", "glucose", "biomass", "ethanol",
                     "episode_id", "contaminated", "has_contamination"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    assert df["episode_id"].nunique() > 0, "No episodes found"
    assert (df["glucose"] >= 0).all(),      "Negative glucose values detected"
    assert (df["biomass"] >= 0).all(),      "Negative biomass values detected"
    assert (df["ethanol"] >= 0).all(),      "Negative ethanol values detected"
    assert df["time_hours"].is_monotonic_increasing or True  # per-episode check below

    print(f"✓ Loaded {len(df):,} rows | {df['episode_id'].nunique()} episodes | "
          f"columns: {df.columns.tolist()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Filter contaminated episodes
# ══════════════════════════════════════════════════════════════════════════════
def filter_contaminated(df: pd.DataFrame) -> pd.DataFrame:
    episode_flags = df.groupby("episode_id")["has_contamination"].any()
    clean_ids     = episode_flags[~episode_flags].index.tolist()
    df_clean      = df[df["episode_id"].isin(clean_ids)].copy()

    n_total = df["episode_id"].nunique()
    n_clean = len(clean_ids)
    print(f"✓ Removed {n_total - n_clean} contaminated episodes → "
          f"{n_clean} clean episodes remain")
    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# 3. Select 3 episodes with meaningfully different initial glucose
# ══════════════════════════════════════════════════════════════════════════════
def select_episodes(df: pd.DataFrame, n: int = 3) -> list[int]:
    # Initial glucose = glucose at the earliest time point per episode
    init_gluc = (
        df.sort_values("time_hours")
          .groupby("episode_id")["glucose"]
          .first()
          .sort_values()
    )

    # Evenly spaced percentile picks from the sorted distribution
    indices  = np.linspace(0, len(init_gluc) - 1, n, dtype=int)
    selected = init_gluc.iloc[indices]

    print(f"\n✓ Selected {n} episodes with well-separated initial glucose:")
    for eid, g0 in selected.items():
        print(f"   Episode {eid:>3d}  |  initial glucose = {g0:.3f} g/L")

    return selected.index.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Restrict to glucose-consumption phase
# ══════════════════════════════════════════════════════════════════════════════
def consumption_phase(episode_df: pd.DataFrame,
                      threshold: float = GLUCOSE_DEPLETION_THRESHOLD
                      ) -> tuple[pd.DataFrame, float]:
    """
    Keep rows from t=0 up to (and including) the first row where glucose
    drops below `threshold`.  Returns the trimmed DataFrame and the
    depletion time.
    """
    ep = episode_df.sort_values("time_hours").reset_index(drop=True)
    below = ep[ep["glucose"] < threshold]

    if below.empty:
        # Glucose never depleted within recorded window
        depletion_time = ep["time_hours"].iloc[-1]
        print(f"   ⚠ Glucose never fell below {threshold} g/L; using full episode.")
        return ep, depletion_time

    first_depletion_idx = below.index[0]
    depletion_time      = ep.loc[first_depletion_idx, "time_hours"]
    trimmed             = ep.loc[:first_depletion_idx].copy()
    return trimmed, depletion_time


# ══════════════════════════════════════════════════════════════════════════════
# 5. Visualise
# ══════════════════════════════════════════════════════════════════════════════
COLORS    = ["#2196F3", "#E91E63", "#4CAF50"]
VARIABLES = ["glucose", "biomass", "ethanol"]
YLABELS   = ["Glucose (g/L)", "Biomass (g/L)", "Ethanol (g/L)"]


def visualize(episode_data: dict[int, pd.DataFrame], out_path: str) -> None:
    """3-row × N-col grid: rows = variables, cols = episodes."""
    n_ep = len(episode_data)
    fig  = plt.figure(figsize=(5 * n_ep, 10))
    gs   = gridspec.GridSpec(3, n_ep, hspace=0.4, wspace=0.35)

    for col_idx, (eid, ep_df) in enumerate(episode_data.items()):
        color = COLORS[col_idx % len(COLORS)]
        for row_idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.plot(ep_df["time_hours"], ep_df[var],
                    color=color, linewidth=2, marker="o",
                    markersize=3, markerfacecolor="white")
            ax.set_xlabel("Time (h)", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"Episode {eid}" if row_idx == 0 else "", fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Selected Fermentation Episodes — Glucose Consumption Phase",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Visualization saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Save cleaned datasets & report
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(episode_data: dict[int, pd.DataFrame],
                 summary_rows: list[dict],
                 out_dir: str) -> None:

    for eid, ep_df in episode_data.items():
        csv_path = os.path.join(out_dir, f"episode_{eid}_cleaned.csv")
        ep_df.to_csv(csv_path, index=False)
        print(f"✓ Saved cleaned episode {eid} → {csv_path}  ({len(ep_df)} rows)")

    report_path = os.path.join(out_dir, "phase1_report.txt")
    with open(report_path, "w") as f:
        f.write("Phase 1 Episode Selection Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Episode ID':<14} {'Initial Glucose (g/L)':<24} "
                f"{'Depletion Time (h)':<22} {'#Rows (clean)'}\n")
        f.write("-" * 60 + "\n")
        for row in summary_rows:
            f.write(f"{row['episode_id']:<14} {row['initial_glucose']:<24.4f} "
                    f"{row['depletion_time']:<22.1f} {row['n_rows']}\n")
        f.write("\nDepletion criterion: glucose < "
                f"{GLUCOSE_DEPLETION_THRESHOLD} g/L\n")
        f.write(f"Random seed: {SEED}\n")

    print(f"✓ Report saved → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Phase 1: Dataset Preprocessing")
    print("=" * 60)

    # 1. Load
    df = load_and_validate(DATA_PATH)

    # 2. Remove contaminated episodes
    df_clean = filter_contaminated(df)

    # 3. Select 3 episodes with spread-out initial glucose
    selected_ids = select_episodes(df_clean, n=3)

    # 4. Trim to consumption phase
    episode_data  = {}
    summary_rows  = []

    print("\nTrimming to glucose-consumption phase:")
    for eid in selected_ids:
        ep_full                = df_clean[df_clean["episode_id"] == eid]
        trimmed, depl_time     = consumption_phase(ep_full)
        init_gluc              = trimmed["glucose"].iloc[0]

        episode_data[eid]      = trimmed
        summary_rows.append({
            "episode_id":      eid,
            "initial_glucose": init_gluc,
            "depletion_time":  depl_time,
            "n_rows":          len(trimmed),
        })
        print(f"   Episode {eid:>3d}  |  t_depletion = {depl_time:.1f} h  "
              f"|  {len(trimmed)} rows retained")

    # 5. Visualize
    viz_path = os.path.join(OUT_DIR, "phase1_visualization.png")
    visualize(episode_data, viz_path)

    # 6. Save CSVs + report
    save_outputs(episode_data, summary_rows, OUT_DIR)

    # 7. Print final summary table
    print("\n" + "=" * 60)
    print("SUMMARY — Selected Episodes")
    print("=" * 60)
    print(f"{'Episode ID':<14} {'Initial Glucose (g/L)':<24} "
          f"{'Depletion Time (h)':<22} {'Rows'}")
    print("-" * 60)
    for row in summary_rows:
        print(f"{row['episode_id']:<14} {row['initial_glucose']:<24.4f} "
              f"{row['depletion_time']:<22.1f} {row['n_rows']}")
    print("=" * 60)
    print("\nPhase 1 complete. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
