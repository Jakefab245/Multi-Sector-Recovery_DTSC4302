"""
=============================================================================
Figure 8 — Disaster Exposure and Labor Market Volatility Across U.S. States
            (2006–2024, COVID Years Excluded)
=============================================================================
Scatter plot of ALL 50 U.S. states:
  X : Total FEMA disaster declarations (2006-2024, excl. 2020-2021)
  Y : Average annual unemployment volatility (std dev of monthly rates)
  Red dots  = 15 high-exposure states (labeled)
  Grey dots = other 35 states

Sources: BLS LAUS; FEMA Disaster Declarations Database.
=============================================================================
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH     = os.path.join(BASE, "FEMA_Disaster_2006_2025", "laus_with_fema_disaster_exposure_2006_2025.csv")
RECOVERY_PATH = os.path.join(BASE, "recovery_speed_results.csv")
OUTPUT        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "Fig8_disaster_vs_volatility.png")

# ── 15 HIGH-EXPOSURE STATES ───────────────────────────────────────────────────
STATES_15 = [
    "Florida", "Tennessee", "Texas", "Missouri", "North Carolina",
    "Arkansas", "Georgia", "Mississippi", "Kansas", "Oklahoma",
    "Kentucky", "Nebraska", "Virginia", "Iowa", "Louisiana",
]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df  = pd.read_csv(DATA_PATH, parse_dates=["date"])
rec = pd.read_csv(RECOVERY_PATH)
abbr_map = rec.set_index("state")["abbr"].to_dict()

# Filter: 2006-2024, exclude 2020-2021
mask = (df["year"] >= 2006) & (df["year"] <= 2024) & (~df["year"].isin([2020, 2021]))
df_f = df[mask].copy()

# ── X: Total FEMA declarations per state ──────────────────────────────────────
total_fema = df_f.groupby("state")["disaster_count_month"].sum().rename("total_fema")

# ── Y: Avg annual unemployment volatility ─────────────────────────────────────
#   For each (state, year): std dev of the 12 monthly rates
#   Then average across years
annual_vol = (
    df_f.groupby(["state", "year"])["unemployment_rate"]
    .std()
    .groupby("state")
    .mean()
    .rename("avg_volatility")
)

# ── MERGE ─────────────────────────────────────────────────────────────────────
scatter = pd.DataFrame({"total_fema": total_fema, "avg_volatility": annual_vol}).dropna()
scatter["is_top15"] = scatter.index.isin(STATES_15)
scatter["abbr"]     = scatter.index.map(abbr_map)

# ── OLS (all 50 states) ──────────────────────────────────────────────────────
slope, intercept, r, p, _ = stats.linregress(
    scatter["total_fema"], scatter["avg_volatility"]
)
r2 = r ** 2

# ── REFERENCE LINES ──────────────────────────────────────────────────────────
avg_top15 = scatter.loc[scatter["is_top15"], "avg_volatility"].mean()
avg_other = scatter.loc[~scatter["is_top15"], "avg_volatility"].mean()

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10.5, 7.5))
fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)

# Grey dots — other states
other = scatter[~scatter["is_top15"]]
ax.scatter(other["total_fema"], other["avg_volatility"],
           s=60, color="#C8C8C8", edgecolors="#aaa", linewidths=0.5,
           zorder=2, alpha=0.85)

# Red dots — top 15
top = scatter[scatter["is_top15"]]
ax.scatter(top["total_fema"], top["avg_volatility"],
           s=90, color="#A93226", edgecolors="white", linewidths=0.7,
           zorder=4, alpha=0.92)

# State labels for top 15
label_nudge = {
    "MS": (-30, 0.012),  "KY": (-30, 0.012),  "MO": (20, 0.008),
    "LA": (-30, 0.008),  "IA": (20, 0.008),   "TN": (-30, -0.020),
    "AR": (-30, -0.020), "NC": (-30, -0.015),  "FL": (20, -0.015),
    "GA": (20, -0.015),  "KS": (20, -0.015),   "OK": (20, -0.015),
    "VA": (-30, -0.015), "NE": (-30, -0.015),  "TX": (20, 0.005),
}
for state, row in top.iterrows():
    ab = row["abbr"]
    dx, dy = label_nudge.get(ab, (20, 0.005))
    ha = "right" if dx < 0 else "left"
    ax.annotate(ab, (row["total_fema"], row["avg_volatility"]),
                xytext=(row["total_fema"] + dx, row["avg_volatility"] + dy),
                fontsize=8.5, fontweight="bold", color="#A93226",
                ha=ha, va="center", zorder=5)

# OLS trend line
x_line = np.linspace(scatter["total_fema"].min() - 30,
                     scatter["total_fema"].max() + 30, 300)
ax.plot(x_line, slope * x_line + intercept,
        color="#555", linewidth=1.5, linestyle="--", alpha=0.7, zorder=1)

# Horizontal reference lines
ax.axhline(avg_top15, color="#C0392B", linewidth=0.8, linestyle=":",
           alpha=0.6, zorder=0)
ax.text(scatter["total_fema"].max() + 15, avg_top15 + 0.005,
        f"Top 15 avg = {avg_top15:.2f}",
        fontsize=7.5, color="#C0392B", va="bottom")

ax.axhline(avg_other, color="#888", linewidth=0.8, linestyle=":",
           alpha=0.5, zorder=0)
ax.text(scatter["total_fema"].max() + 15, avg_other + 0.005,
        f"Other states avg = {avg_other:.2f}",
        fontsize=7.5, color="#888", va="bottom")

# OLS stats box
ax.text(0.025, 0.97,
        f"OLS Trendline (50 states)\n"
        f"$r^2$ = {r2:.2f}  |  slope = {slope:.6f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f5",
                  edgecolor="#bbb", alpha=0.95), zorder=10)

# ── LEGEND ───────────────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#C8C8C8",
                  markersize=8, label="Other U.S. States", markeredgecolor="#aaa"),
    mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#A93226",
                  markersize=9, label="15 High-Exposure States",
                  markeredgecolor="white"),
    mlines.Line2D([], [], color="#555", linewidth=1.5, linestyle="--",
                  label="OLS Trendline (all states)"),
]
ax.legend(handles=handles, loc="upper right", fontsize=9,
          framealpha=0.95, edgecolor="#ccc")

# ── AXES ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Total FEMA Disaster Declarations (2006–2024, excl. 2020–2021)",
              fontsize=10.5, labelpad=8)
ax.set_ylabel("Average Annual Unemployment Volatility\n"
              "(Std Dev of Monthly Rates)",
              fontsize=10.5, labelpad=8)
ax.set_title("Disaster Exposure and Labor Market Volatility Across U.S. States\n"
             "(2006–2024, COVID Years Excluded)",
             fontsize=13, fontweight="bold", pad=12)
ax.tick_params(labelsize=9)
ax.grid(True, alpha=0.25, linewidth=0.5, color="#bbb")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── INTERPRETATION NOTE ──────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Note: Weak negative slope reflects that high-exposure states tend to be "
         "large, economically diversified states with greater shock-absorption capacity.",
         ha="center", fontsize=8, color="#555", style="italic")

# ── CAPTION (below note) ─────────────────────────────────────────────────────
fig.text(0.5, -0.04,
         "Figure 8. Disaster Exposure and Labor Market Volatility Across U.S. States "
         "(2006–2024, excluding COVID years).\n"
         "Source: BLS LAUS; FEMA Disaster Declarations.",
         ha="center", fontsize=9, color="#333", style="italic",
         fontweight="bold")

# ── SOURCE (inside axes) ────────────────────────────────────────────────────
ax.text(0.99, 0.01,
        "Source: BLS LAUS; FEMA Disaster Declarations. 2006–2024.",
        transform=ax.transAxes, fontsize=7.5, color="#888",
        ha="right", va="bottom", style="italic")

# ── SAVE & SHOW ──────────────────────────────────────────────────────────────
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white",
            pad_inches=0.4)
plt.show()
plt.close()

print(f"\n✓ Saved: {OUTPUT}")
print(f"  OLS: slope={slope:.6f} | r²={r2:.3f} | p={p:.4f}")
print(f"  Top 15 avg volatility: {avg_top15:.3f}")
print(f"  Other states avg volatility: {avg_other:.3f}")
print(f"  Total states plotted: {len(scatter)}")
