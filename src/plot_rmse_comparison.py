import matplotlib.pyplot as plt
import os

# ============================================================
# RESULTS
# ============================================================

methods = [
    "PPO (7)",
    "DQN (7)",
    "Even (7)",
    "Even (9)",
    "Random (7)",
    "Random (12.3)"
]

holes = [
    7,
    7,
    7,
    9,
    7,
    12.33
]

rmse = [
    0.88195,
    0.90725,
    1.01811,
    0.79274,
    1.64757,
    0.77764
]

# ============================================================
# FIGURE 1: RMSE COMPARISON
# ============================================================

pairs = sorted(zip(rmse, methods))

rmse_sorted = [p[0] for p in pairs]
methods_sorted = [p[1] for p in pairs]

colors_bar = [
    "#2ca02c",
    "#98df8a",
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#d62728"
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    methods_sorted,
    rmse_sorted,
    color=colors_bar
)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.03,
        f"{height:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title(
    "RMSE Comparison of Drilling Strategies\n(Lower RMSE is Better)",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Method")
plt.ylabel("Mean RMSE")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2: ACCURACY VS DRILLING COST
# ============================================================

colors_scatter = [
    "green",       # PPO
    "royalblue",   # DQN
    "orange",      # Even 7
    "gold",        # Even 9
    "red",         # Random 7
    "darkgreen"    # Random 12
]

plt.figure(figsize=(10, 6))

plt.scatter(
    holes,
    rmse,
    c=colors_scatter,
    s=280,
    edgecolors="black",
    linewidths=1.2
)

for i in range(len(methods)):

    plt.annotate(
        methods[i],
        (holes[i], rmse[i]),
        xytext=(8, 5),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

# 7-hole budget reference line
plt.axvline(
    x=7,
    color="gray",
    linestyle="--",
    alpha=0.6,
    label="7-hole budget"
)

plt.title(
    "Accuracy vs Drilling Cost",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Number of Boreholes")
plt.ylabel("Mean RMSE")

plt.grid(
    linestyle="--",
    alpha=0.4
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "holes_vs_rmse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# OPEN IMAGES AUTOMATICALLY (MACOS)
# ============================================================

print("\nGenerated:")
print(" - rmse_comparison.png")
print(" - holes_vs_rmse.png")

os.system("open rmse_comparison.png")
os.system("open holes_vs_rmse.png")
