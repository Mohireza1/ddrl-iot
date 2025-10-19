import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] CSV not found: {path}", file=sys.stderr)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _plot_xy(x, y, xlabel, ylabel, title, outfile: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"[OK] Saved {outfile}")


def _plot_xy_multi(x, curves, labels, xlabel, ylabel, title, outfile: Path):
    plt.figure(figsize=(10, 6))
    for y, lbl in zip(curves, labels):
        plt.plot(x, y, label=lbl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="major", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"[OK] Saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, default=Path("runs/episode_summaries.csv"))
    ap.add_argument("--perstep", type=Path, default=Path("runs/per_step.csv"))
    ap.add_argument("--out", type=Path, default=Path("runs/plots"))
    args = ap.parse_args()

    df_ep = _safe_read_csv(args.episodes)
    df_ps = _safe_read_csv(args.perstep)
    _ensure_outdir(args.out)

    # 1) Total reward vs episode
    if {"episode", "reward_sum"}.issubset(df_ep.columns):
        _plot_xy(
            df_ep["episode"].values,
            df_ep["reward_sum"].values,
            "Episode",
            "Total reward",
            "Total reward vs episode",
            args.out / "total_reward_vs_episode.png",
        )

    # 2) Min/avg/max reward vs episode
    if {"episode", "reward_mean", "reward_min", "reward_max"}.issubset(df_ep.columns):
        _plot_xy_multi(
            df_ep["episode"].values,
            [
                df_ep["reward_min"].values,
                df_ep["reward_mean"].values,
                df_ep["reward_max"].values,
            ],
            ["min", "mean", "max"],
            "Episode",
            "Reward",
            "Min/Mean/Max reward per episode",
            args.out / "reward_min_mean_max_vs_episode.png",
        )

    # 3) Loss vs episode
    if not df_ps.empty and {"episode", "loss"}.issubset(df_ps.columns):
        grp = df_ps.groupby("episode", as_index=False)["loss"].mean(numeric_only=True)
        _plot_xy(
            grp["episode"].values,
            grp["loss"].values,
            "Episode",
            "Average loss",
            "Loss vs episode",
            args.out / "loss_vs_episode.png",
        )

    # 4) MEC tasks percentage vs episode
    # Prefer the paper definition: ratio-of-sums of (mec_bits / arr_bits) per episode.
    if not df_ps.empty and {"episode", "mec_bits", "arr_bits"}.issubset(df_ps.columns):
        g = df_ps.groupby("episode", as_index=False)[["mec_bits", "arr_bits"]].sum(
            numeric_only=True
        )
        y = 100.0 * g["mec_bits"] / np.maximum(g["arr_bits"], 1e-12)
        _plot_xy(
            g["episode"].values,
            y.values,
            "Episode",
            "Avg MEC task percentage (%)",
            "Avg % of tasks completed by MEC (per episode)",
            args.out / "mec_percentage_vs_episode.png",
        )
    # Fallback: ratio-of-sums of served work (diagnostic)
    elif not df_ps.empty and {
        "episode",
        "mec_served_bits",
        "local_served_bits",
    }.issubset(df_ps.columns):
        g = df_ps.groupby("episode", as_index=False)[
            ["mec_served_bits", "local_served_bits"]
        ].sum(numeric_only=True)
        total = np.maximum(g["mec_served_bits"] + g["local_served_bits"], 1e-12)
        y = 100.0 * g["mec_served_bits"] / total
        _plot_xy(
            g["episode"].values,
            y.values,
            "Episode",
            "Avg MEC task percentage (%)",
            "Avg % of served work at MEC (per episode)",
            args.out / "mec_percentage_vs_episode.png",
        )
    # Last resort: mean of per-step percentages
    elif not df_ps.empty and {"episode", "mec_task_pct"}.issubset(df_ps.columns):
        g = df_ps.groupby("episode", as_index=False)["mec_task_pct"].mean(
            numeric_only=True
        )
        _plot_xy(
            g["episode"].values,
            g["mec_task_pct"].values,
            "Episode",
            "Avg MEC task percentage (%)",
            "Avg % of tasks completed by MEC (per episode) [mean of per-step %]",
            args.out / "mec_percentage_vs_episode.png",
        )
    else:
        print("[WARN] No MEC metrics present in per_step.csv", file=sys.stderr)

    # 5) Also emit a separate diagnostic plot for served-share when available
    if not df_ps.empty and {"episode", "mec_served_bits", "local_served_bits"}.issubset(
        df_ps.columns
    ):
        g2 = df_ps.groupby("episode", as_index=False)[
            ["mec_served_bits", "local_served_bits"]
        ].sum(numeric_only=True)
        total2 = np.maximum(g2["mec_served_bits"] + g2["local_served_bits"], 1e-12)
        y2 = 100.0 * g2["mec_served_bits"] / total2
        _plot_xy(
            g2["episode"].values,
            y2.values,
            "Episode",
            "MEC served share (%)",
            "MEC share of served work (per episode)",
            args.out / "mec_served_share_vs_episode.png",
        )


if __name__ == "__main__":
    main()
