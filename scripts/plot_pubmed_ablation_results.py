"""Plot the archived PubMed architecture-ablation ROUGE results."""

import matplotlib.pyplot as plt


CONFIGURATIONS = ("Layer 1 (1-gram)", "Stacked (1-2)", "Stacked (1-3)", "Multi-Kernel (1-4)")
ROUGE_1_SCORES = (0.6262, 0.6262, 0.6780, 0.7143)
OUTPUT_PATH = "ablation_results_graph.png"


def main() -> None:
    """Render the archived ablation plot without mutating source data."""
    _, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        CONFIGURATIONS,
        ROUGE_1_SCORES,
        marker="o",
        color="#1f77b4",
        linewidth=2.5,
        markersize=8,
        label="ROUGE-1",
    )
    axis.set_title("Ablation Study: Stacked CNN Kernel Progression", fontsize=12, fontweight="bold", pad=12)
    axis.set_xlabel("Configurations", fontsize=10)
    axis.set_ylabel("ROUGE-1 Score", fontsize=10)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.set_ylim(0.60, 0.73)
    axis.annotate(
        "Proposed (0.7143)",
        xy=(3, 0.7143),
        xytext=(2.1, 0.65),
        arrowprops={"facecolor": "black", "arrowstyle": "->", "lw": 1},
        fontsize=9,
        fontweight="semibold",
        backgroundcolor="white",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
