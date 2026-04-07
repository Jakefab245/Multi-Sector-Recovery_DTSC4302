# Python code content for volatility_timeline_with_disasters.py


"""
=============================================================================
FIG 4: Annotated Volatility Timeline with FEMA Declarations Bar Chart
=============================================================================
Two-panel figure:
  - Top: Mean unemployment volatility across top 15 states with IQR band
         and major disaster events annotated
  - Bottom: Total FEMA declarations per year (bar chart)

Data required:
  - laus_with_fema_disaster_exposure_2006_2025.csv
    Columns: state, date, year, month, unemployment_rate,
             disaster_count_month, disaster_exposure_12m

Output: fig4_timeline.png (300 DPI)
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ──
DATA_PATH = "FEMA_Disaster_2006_2025/laus_with_fema_disaster_exposure_2006_2025.csv"
OUTPUT_PATH = "fig4_timeline.png"

TOP_15 = [
    "Texas", "Kentucky", "Missouri", "Oklahoma", "Georgia",
    "Louisiana", "Florida", "Kansas", "North Carolina", "Virginia",
    "Mississippi", "Nebraska", "Iowa", "Arkansas", "Tennessee"
]

# Major disaster events to annotate on the timeline
# Format: (year, label, position: "above" or "below")
ANNOTATIONS = [
    (2008, "Great Recession\n+ Iowa/Gulf storms", "above"),
    (2011, "Record tornado\nseason (1,700+)", "above"),
    (2013, "OK tornadoes\n+ Midwest floods", "below"),
    (2017, "Harvey, Irma,\nMaria season", "above"),
    (2020, "COVID-19\npandemic", "above"),
    (2021, "Ida, winter storms\n+ recovery lag", "below"),
    (2024, "Helene, Milton\n+ severe storms", "below"),
]


# ── LOAD DATA ──
df = pd.read_csv(DATA_PATH)
df_top = df[df["state"].isin(TOP_15)].copy()


# ── COMPUTE VOLATILITY (std dev of monthly unemployment per state-year) ──
volatility = (
    df_top.groupby(["state", "year"])["unemployment_rate"]
    .std()
    .reset_index()
    .rename(columns={"unemployment_rate": "volatility"})
)

# Yearly statistics across the 15 states
yearly_stats = volatility.groupby("year")["volatility"].agg(
    ["mean", "median", "std"]
).reset_index()
yearly_stats.columns = ["year", "mean", "median", "std"]

# IQR (25th and 75th percentiles)
yearly_pct = volatility.groupby("year")["volatility"].quantile([0.25, 0.75]).unstack()
yearly_pct.columns = ["p25", "p75"]
yearly_stats = yearly_stats.merge(yearly_pct, left_on="year", right_index=True)

# Min/max for outer band
yearly_minmax = volatility.groupby("year")["volatility"].agg(["min", "max"]).reset_index()
yearly_stats = yearly_stats.merge(yearly_minmax, on="year")


# ── COMPUTE TOTAL FEMA DECLARATIONS PER YEAR ──
disaster_yearly = (
    df_top.groupby("year")["disaster_count_month"]
    .sum()
    .reset_index()
    .rename(columns={"disaster_count_month": "total_dec"})
)


# ── BUILD FIGURE ──
fig, (ax_main, ax_bar) = plt.subplots(
    2, 1, figsize=(14, 9),
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    sharex=True
)

years = yearly_stats["year"].values
mean_vol = yearly_stats["mean"].values
p25 = yearly_stats["p25"].values
p75 = yearly_stats["p75"].values
ymin = yearly_stats["min"].values
ymax = yearly_stats["max"].values


# ── TOP PANEL: Volatility timeline ──

# Outer band (full range across states)
ax_main.fill_between(years, ymin, ymax, alpha=0.06, color="#1565C0",
                      label="Full range across 15 states")

# IQR band (25th-75th percentile)
ax_main.fill_between(years, p25, p75, alpha=0.18, color="#1565C0",
                      label="Interquartile range (25th–75th percentile)")

# Mean volatility line with markers
ax_main.plot(years, mean_vol, color="#1565C0", linewidth=2.5, zorder=3,
             marker="o", markersize=6, markerfacecolor="white",
             markeredgecolor="#1565C0", markeredgewidth=1.5,
             label="Mean volatility across top 15 states")

# Value labels on key spike years
for y in [2008, 2011, 2017, 2020, 2021]:
    idx = np.where(years == y)[0][0]
    val = mean_vol[idx]
    ax_main.text(y, val + 0.08, f"{val:.2f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#1565C0")

# Disaster event annotations
for yr, text, pos in ANNOTATIONS:
    idx = np.where(years == yr)[0]
    if len(idx) == 0:
        continue
    idx = idx[0]
    val = mean_vol[idx]

    if pos == "above":
        text_y = val + 0.35 + (0.6 if yr == 2020 else 0)
        va = "bottom"
    else:
        text_y = val - 0.25
        va = "top"

    ax_main.annotate(
        text, xy=(yr, val), xytext=(yr, text_y),
        fontsize=8.5, color="#333", ha="center", va=va,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor="#E53935", alpha=0.9, linewidth=1.2),
        arrowprops=dict(arrowstyle="-|>", color="#E53935",
                       lw=1.2, connectionstyle="arc3,rad=0"),
    )

# COVID year shading
ax_main.axvspan(2019.5, 2020.5, alpha=0.08, color="#E53935", zorder=0)

# Overall mean reference line
overall_mean = np.mean(mean_vol)
ax_main.axhline(y=overall_mean, color="#999", linewidth=0.8, linestyle=":",
                alpha=0.5, zorder=1)
ax_main.text(2025.3, overall_mean, f"Mean: {overall_mean:.2f}",
             fontsize=8, color="#999", va="center")

# Formatting
ax_main.set_ylabel("Unemployment Volatility\n(Std. Dev. of Monthly Rate)",
                    fontsize=11, labelpad=10)
ax_main.set_ylim(0, max(mean_vol) + 1.0)
ax_main.grid(axis="y", alpha=0.2, linewidth=0.5)
ax_main.grid(axis="x", alpha=0.1, linewidth=0.3)
ax_main.spines["top"].set_visible(False)
ax_main.spines["right"].set_visible(False)
ax_main.tick_params(labelsize=10)
ax_main.legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor="#ddd")

ax_main.set_title(
    "Unemployment Volatility Across High-Exposure U.S. States (2006–2025)\n"
    "with Major Disaster Events Annotated",
    fontsize=14, fontweight="bold", pad=40
)


# ── BOTTOM PANEL: FEMA declarations bar chart ──
dec_values = disaster_yearly.set_index("year").reindex(years)["total_dec"].values

# Color bars by intensity
bar_colors = []
for d in dec_values:
    if d > 1500:
        bar_colors.append("#C62828")      # Very high
    elif d > 800:
        bar_colors.append("#E53935")      # High
    elif d > 400:
        bar_colors.append("#FF8A65")      # Moderate
    else:
        bar_colors.append("#FFCCBC")      # Low

ax_bar.bar(years, dec_values, color=bar_colors, edgecolor="white",
           linewidth=0.5, width=0.8)

# Value labels on bars
for yr, dec in zip(years, dec_values):
    if dec > 200:
        ax_bar.text(yr, dec + 30, f"{int(dec)}", ha="center", va="bottom",
                    fontsize=7, color="#555")

ax_bar.set_ylabel("Total FEMA\nDeclarations", fontsize=10, labelpad=10)
ax_bar.set_xlabel("Year", fontsize=11)
ax_bar.set_xticks(years)
ax_bar.set_xticklabels(years, fontsize=9, rotation=45, ha="right")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.tick_params(labelsize=9)
ax_bar.grid(axis="y", alpha=0.15, linewidth=0.5)




# ── SAVE ──
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.4)
plt.show()
plt.close()
print(f"✓ Saved {OUTPUT_PATH}")
print(f"  Overall mean volatility: {overall_mean:.2f}")
print(f"  Peak year: 2020 ({mean_vol[np.where(years==2020)[0][0]]:.2f})")
print(f"  Top 5 years by volatility: {', '.join(str(y) for y in years[np.argsort(mean_vol)[-5:][::-1]])}")

