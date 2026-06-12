from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def configure_times() -> font_manager.FontProperties | None:
    times_path = Path("C:/Windows/Fonts/times.ttf")
    times_prop = None
    if times_path.exists():
        font_manager.fontManager.addfont(str(times_path))
        times_prop = font_manager.FontProperties(fname=str(times_path))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return times_prop


def apply_times(ax, times_prop: font_manager.FontProperties | None) -> None:
    texts = (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    )
    for text in texts:
        if times_prop is not None:
            text.set_fontproperties(times_prop)
        else:
            text.set_fontname("Times New Roman")


def pct(values: pd.Series | np.ndarray) -> np.ndarray:
    return 100.0 * np.asarray(values, dtype=float)


def main() -> None:
    times_prop = configure_times()

    reliability = pd.read_csv(TABLES / "vacs_validation_size_reliability_summary.csv")
    repeated = pd.read_csv(TABLES / "vacs_repeated_validation_summary.csv")

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.75, 2.85),
        gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.28},
    )
    fig.suptitle(
        "Protocol robustness under validation choice and repetition",
        y=0.995,
        fontsize=13,
        fontproperties=times_prop,
    )

    labels = [f"{int(round(frac * 100))}%" for frac in reliability["validation_fraction"]]
    x = np.arange(len(labels))
    gain_points = pct(reliability["gain"])
    agreement = pct(reliability["top1_agreement"])

    ax_left.bar(
        x,
        gain_points,
        width=0.46,
        color="#4C78A8",
        edgecolor="#222222",
        linewidth=0.7,
        label="Gain vs. herding",
    )
    ax_left.axhline(0.0, color="#666666", linewidth=0.8)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)
    ax_left.set_xlabel("Validation fraction", fontproperties=times_prop)
    ax_left.set_ylabel("Gain (points)", fontproperties=times_prop)
    ax_left.set_title("Validation-size reliability", fontproperties=times_prop, pad=7)
    ax_left.set_ylim(-1.45, 1.55)
    ax_left.grid(axis="y", alpha=0.25)

    ax_agree = ax_left.twinx()
    ax_agree.plot(
        x,
        agreement,
        color="#F58518",
        marker="o",
        linewidth=1.8,
        label="Top-1 agreement",
    )
    ax_agree.set_ylabel("Top-1 agreement (%)", fontproperties=times_prop)
    ax_agree.set_ylim(30, 84)
    apply_times(ax_agree, times_prop)

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_agree.get_legend_handles_labels()
    legend_left = ax_left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        prop=times_prop,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        borderaxespad=0.0,
    )
    for text in legend_left.get_texts():
        if times_prop is not None:
            text.set_fontproperties(times_prop)

    slice_order = ["Pooled", "20NG", "Breast cancer", "Digits", "Iris", "Wine"]
    repeated = repeated.set_index("slice").loc[slice_order].reset_index()
    x2 = np.arange(len(slice_order))
    width = 0.34
    default = pct(repeated["default_accuracy"])
    repeated_acc = pct(repeated["repeated_accuracy"])

    ax_right.bar(
        x2 - width / 2,
        default,
        width=width,
        color="#54A24B",
        edgecolor="#222222",
        linewidth=0.65,
        label="VACS-F",
    )
    ax_right.bar(
        x2 + width / 2,
        repeated_acc,
        width=width,
        color="#E45756",
        edgecolor="#222222",
        linewidth=0.65,
        label="VACS-R",
    )
    ax_right.set_title("Repeated internal validation", fontproperties=times_prop, pad=7)
    ax_right.set_ylabel("Accuracy (%)", fontproperties=times_prop)
    ax_right.set_ylim(0, 96)
    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(["Pooled", "20NG", "Breast\ncancer", "Digits", "Iris", "Wine"])
    ax_right.grid(axis="y", alpha=0.25)
    pooled_delta = repeated_acc[0] - default[0]
    ax_right.text(
        x2[0],
        max(default[0], repeated_acc[0]) + 2.0,
        f"+{pooled_delta:.2f} pts",
        ha="center",
        va="bottom",
        fontsize=9,
        fontproperties=times_prop,
    )
    legend_right = ax_right.legend(
        frameon=False,
        prop=times_prop,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0.0,
        handlelength=1.6,
        ncol=2,
    )
    for text in legend_right.get_texts():
        if times_prop is not None:
            text.set_fontproperties(times_prop)

    apply_times(ax_left, times_prop)
    apply_times(ax_right, times_prop)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.29, top=0.78, wspace=0.30)
    fig.savefig(FIGURES / "vacs_protocol_robustness.pdf")
    fig.savefig(FIGURES / "vacs_protocol_robustness.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
