"""Research design figure matching the thesis-core flow."""

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from experiments.lib import RESULTS


def box(ax, x, y, w, h, text):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1,
        edgecolor="#1565c0",
        facecolor="#e3f2fd",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#37474f"),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    box(ax, 0.2, 1.6, 1.7, 0.8, "Data audit")
    box(ax, 2.6, 2.8, 1.9, 0.8, "RQ1\nPhase effect")
    box(ax, 2.6, 1.6, 1.7, 0.8, "Models")
    box(ax, 5.0, 2.8, 1.9, 0.8, "RQ2\nDrivers by phase")
    box(ax, 5.0, 1.6, 1.9, 0.8, "RQ3\nFuture period")
    box(ax, 5.0, 0.4, 1.9, 0.8, "RQ4\nCalibration")
    box(ax, 7.6, 0.4, 1.4, 0.8, "DSS")
    box(ax, 9.3, 0.4, 1.5, 0.8, "Counselor\nevaluation")
    arrow(ax, 1.9, 2.1, 2.6, 3.1)
    arrow(ax, 1.9, 2.0, 2.6, 2.0)
    arrow(ax, 4.3, 2.1, 5.0, 3.1)
    arrow(ax, 4.3, 2.0, 5.0, 2.0)
    arrow(ax, 4.3, 1.9, 5.0, 0.9)
    arrow(ax, 6.9, 0.8, 7.6, 0.8)
    arrow(ax, 9.0, 0.8, 9.3, 0.8)
    ax.set_title("Capstone C research design", loc="left", fontsize=12)
    fig.tight_layout()
    out = RESULTS.parents[1] / "docs" / "research_design.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS / "research_design.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
