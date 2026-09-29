from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"


def configure_fonts():
    times_path = Path("C:/Windows/Fonts/times.ttf")
    times_prop = None
    if times_path.exists():
        font_manager.fontManager.addfont(str(times_path))
        times_prop = font_manager.FontProperties(fname=str(times_path))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return times_prop


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    times_prop,
    facecolor="#F8F8F8",
    edgecolor="#202020",
    linestyle="-",
    fontsize=8.2,
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
            linestyle=linestyle,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontproperties=times_prop,
        linespacing=1.12,
    )


def arrow(ax, x1, y1, x2, y2, *, rad=0.0, color="#202020", linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            linewidth=1.05,
            color=color,
            linestyle=linestyle,
            mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    times_prop = configure_fonts()

    fig, ax = plt.subplots(figsize=(7.2, 2.35))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.25)
    ax.axis("off")

    edge = "#202020"
    neutral = "#F8F8F8"
    pool = "#EAF2F8"
    val = "#FDF0D5"
    selector = "#E9F5EC"
    rebuild = "#F4ECF7"

    add_box(ax, 0.15, 2.55, 1.25, 0.72, "Train\nset D", times_prop=times_prop, facecolor=neutral, edgecolor=edge)
    add_box(ax, 1.95, 3.05, 1.45, 0.68, "Selector\npool B", times_prop=times_prop, facecolor=pool, edgecolor=edge)
    add_box(ax, 1.95, 2.05, 1.45, 0.68, "Internal\nvalidation V", times_prop=times_prop, facecolor=val, edgecolor=edge)

    selector_x = 4.15
    selector_y = [3.32, 2.72, 2.12, 1.52]
    selector_labels = [
        "Random",
        "Herding / k-means",
        "K-center / MARC",
        "Boundary",
    ]
    for y, label in zip(selector_y, selector_labels):
        add_box(ax, selector_x, y, 1.65, 0.42, label, times_prop=times_prop, facecolor=selector, edgecolor=edge, fontsize=7.5)

    add_box(
        ax,
        6.55,
        2.36,
        1.55,
        0.82,
        "Train classifier h on\ncandidate coresets",
        times_prop=times_prop,
        facecolor=neutral,
        edgecolor=edge,
        fontsize=7.8,
    )
    add_box(
        ax,
        8.75,
        2.34,
        1.45,
        0.86,
        "Score on V\nchoose a*",
        times_prop=times_prop,
        facecolor=val,
        edgecolor=edge,
        fontsize=7.9,
    )
    add_box(
        ax,
        10.75,
        2.36,
        1.25,
        0.82,
        "Rebuild on D\nfinal coreset",
        times_prop=times_prop,
        facecolor=rebuild,
        edgecolor=edge,
        fontsize=7.7,
    )
    add_box(
        ax,
        8.75,
        0.48,
        1.45,
        0.58,
        "Held-out test\nnever selects a*",
        times_prop=times_prop,
        facecolor="#FFFFFF",
        edgecolor="#777777",
        fontsize=7.6,
        linestyle=(0, (4, 2)),
    )
    add_box(
        ax,
        4.0,
        0.46,
        3.95,
        0.62,
        "No test labels, no hindsight selector tuning",
        times_prop=times_prop,
        facecolor="#FFFFFF",
        edgecolor="#777777",
        fontsize=7.6,
        linestyle=(0, (3, 2)),
    )

    arrow(ax, 1.40, 2.91, 1.95, 3.39)
    arrow(ax, 1.40, 2.91, 1.95, 2.39)
    arrow(ax, 3.40, 3.39, selector_x, 3.53)
    arrow(ax, 3.40, 3.39, selector_x, 2.93, rad=-0.08)
    arrow(ax, 3.40, 3.39, selector_x, 2.33, rad=-0.13)
    arrow(ax, 3.40, 3.39, selector_x, 1.73, rad=-0.18)
    for y in selector_y:
        arrow(ax, selector_x + 1.65, y + 0.21, 6.55, 2.77, rad=0.08)
    arrow(ax, 8.10, 2.77, 8.75, 2.77)
    arrow(ax, 10.20, 2.77, 10.75, 2.77)
    arrow(ax, 9.48, 2.34, 9.48, 1.06, color="#777777", linestyle=(0, (4, 2)))

    ax.text(2.68, 3.86, "stratified 75/25 split", ha="center", va="center", fontsize=7.4, fontproperties=times_prop)
    ax.text(
        4.98,
        1.30,
        "Selector portfolio A is evaluated only inside training data",
        ha="center",
        va="center",
        fontsize=7.6,
        fontproperties=times_prop,
    )

    fig.tight_layout(pad=0.08)
    fig.savefig(FIGURES / "vacs_workflow.pdf")
    fig.savefig(FIGURES / "vacs_workflow.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
