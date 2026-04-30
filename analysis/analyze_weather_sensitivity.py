from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from model.helpers import load_country_timeseries
from model.scenarios import FILE_PATHS


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETWORK_DIR = PROJECT_ROOT / "results" / "networks"
OUTPUT_DIR = PROJECT_ROOT / "results" / "weather_sensitivity_analysis"

TIMESERIES_FILE = PROJECT_ROOT / FILE_PATHS["timeseries_file"]

YEARS = [2015, 2016, 2017, 2018, 2019]
COUNTRY_CODE = "DK"

NETWORK_NAME_TEMPLATE = "base_DK_{year}.nc"

REPORT_FONT_SIZE = 14

CARRIER_ALIASES = {
    "solar": ["solar"],
    "onshore wind": ["onwind", "onshore wind", "wind_onshore", "onshore"],
    "offshore wind": ["offwind", "offshore wind", "wind_offshore", "offshore"],
    "CCGT": ["gas CCGT", "CCGT", "ccgt", "gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
}


# =========================================================
# PLOT STYLE
# =========================================================

def set_report_plot_style() -> None:
    """
    Set basic report-ready matplotlib style.
    """
    plt.rcParams.update(
        {
            "font.size": REPORT_FONT_SIZE,
            "axes.labelsize": REPORT_FONT_SIZE,
            "xtick.labelsize": REPORT_FONT_SIZE,
            "ytick.labelsize": REPORT_FONT_SIZE,
            "legend.fontsize": REPORT_FONT_SIZE,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """
    Save figure as PNG and PDF.
    """
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


# =========================================================
# HELPERS
# =========================================================

def ensure_output_dir(path: Path) -> None:
    """
    Create output directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def get_network_path(network_dir: Path, year: int) -> Path:
    """
    Return the expected solved network path for one weather year.
    """
    return network_dir / NETWORK_NAME_TEMPLATE.format(year=year)


# =========================================================
# CAPACITY ANALYSIS
# =========================================================

def extract_generator_capacities(
    network_path: Path,
    carrier_aliases: dict[str, list[str]],
) -> pd.Series:
    """
    Extract optimized installed generator capacities grouped into report-friendly
    technology categories.
    """
    network = pypsa.Network(network_path)

    if network.generators.empty:
        return pd.Series(
            {technology: 0.0 for technology in carrier_aliases},
            dtype=float,
        )

    if "p_nom_opt" not in network.generators.columns:
        raise KeyError(
            f"'p_nom_opt' not found in generators of {network_path}. "
            "Make sure the network was optimized before saving."
        )

    capacities_by_carrier = network.generators.groupby("carrier")["p_nom_opt"].sum()

    extracted_capacities = {}

    for technology, aliases in carrier_aliases.items():
        matching_aliases = [
            alias for alias in aliases if alias in capacities_by_carrier.index
        ]

        if matching_aliases:
            extracted_capacities[technology] = capacities_by_carrier.loc[
                matching_aliases
            ].sum()
        else:
            extracted_capacities[technology] = 0.0

    return pd.Series(extracted_capacities, dtype=float)


def build_capacity_table(
    network_dir: Path,
    years: list[int],
    carrier_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Build a table of optimized installed capacities by weather year.
    """
    rows = []

    for year in years:
        network_path = get_network_path(
            network_dir=network_dir,
            year=year,
        )

        if not network_path.exists():
            raise FileNotFoundError(
                f"Network file not found: {network_path}\n"
                f"Expected filename pattern: {NETWORK_NAME_TEMPLATE}\n"
                "Make sure you have run and saved all weather-year networks first."
            )

        capacities = extract_generator_capacities(
            network_path=network_path,
            carrier_aliases=carrier_aliases,
        )

        capacities.name = year
        rows.append(capacities)

    capacity_df = pd.DataFrame(rows)
    capacity_df.index.name = "weather_year"

    return capacity_df


# =========================================================
# CAPACITY FACTOR ANALYSIS
# =========================================================

def build_cf_summary(
    timeseries_file: str,
    country_code: str,
    years: list[int],
) -> pd.DataFrame:
    """
    Build yearly summary statistics for renewable capacity factors and load.

    The function uses the shared model helper, so it follows the same logic as
    the model:
    - electricity demand is fixed to 2016 and mapped to the selected year
    - renewable capacity factors use the selected weather year
    """
    rows = []

    for year in years:
        timeseries_data = load_country_timeseries(
            timeseries_file=timeseries_file,
            country_code=country_code,
            year=str(year),
        )

        rows.append(
            {
                "weather_year": year,
                "solar_cf_mean": timeseries_data["solar_cf"].mean(),
                "onshore_wind_cf_mean": timeseries_data["onshore_wind_cf"].mean(),
                "offshore_wind_cf_mean": timeseries_data["offshore_wind_cf"].mean(),
                "wind_advantage": (
                    timeseries_data["offshore_wind_cf"].mean()
                    - timeseries_data["onshore_wind_cf"].mean()
                ),
                "load_mean_mw": timeseries_data["load"].mean(),
                "load_peak_mw": timeseries_data["load"].max(),
            }
        )

    cf_df = pd.DataFrame(rows).set_index("weather_year")

    return cf_df


def build_cf_timeseries_long(
    timeseries_file: str,
    country_code: str,
    years: list[int],
) -> pd.DataFrame:
    """
    Build long-format hourly renewable capacity factor data for all years.
    """
    rows = []

    for year in years:
        timeseries_data = load_country_timeseries(
            timeseries_file=timeseries_file,
            country_code=country_code,
            year=str(year),
        )

        technology_mapping = {
            "solar": timeseries_data["solar_cf"],
            "onshore wind": timeseries_data["onshore_wind_cf"],
            "offshore wind": timeseries_data["offshore_wind_cf"],
        }

        for technology, series in technology_mapping.items():
            valid_series = series.dropna()

            if valid_series.empty:
                continue

            rows.append(
                pd.DataFrame(
                    {
                        "weather_year": year,
                        "technology": technology,
                        "capacity_factor": valid_series.values,
                    }
                )
            )

    if not rows:
        raise ValueError("No capacity factor data could be assembled.")

    cf_long_df = pd.concat(rows, ignore_index=True)

    return cf_long_df


# =========================================================
# METRICS
# =========================================================

def compute_capacity_sensitivity_metrics(
    capacity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute simple variability metrics for installed capacities across weather years.
    """
    metrics = pd.DataFrame(index=capacity_df.columns)

    metrics["mean_capacity_mw"] = capacity_df.mean(axis=0)
    metrics["median_capacity_mw"] = capacity_df.median(axis=0)
    metrics["std_capacity_mw"] = capacity_df.std(axis=0)
    metrics["min_capacity_mw"] = capacity_df.min(axis=0)
    metrics["max_capacity_mw"] = capacity_df.max(axis=0)
    metrics["range_capacity_mw"] = (
        metrics["max_capacity_mw"] - metrics["min_capacity_mw"]
    )

    mean_nonzero = metrics["mean_capacity_mw"].replace(0.0, pd.NA)
    metrics["relative_range"] = metrics["range_capacity_mw"] / mean_nonzero

    return metrics.sort_values("range_capacity_mw", ascending=False)


def build_comparison_table(
    capacity_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine yearly installed capacities and yearly CF summary values.
    """
    return capacity_df.join(cf_df, how="left")


def compute_cf_capacity_correlations(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute simple correlations between yearly mean CF and installed capacity
    for the renewable technologies.
    """
    pairs = [
        ("solar", "solar_cf_mean"),
        ("onshore wind", "onshore_wind_cf_mean"),
        ("offshore wind", "offshore_wind_cf_mean"),
    ]

    rows = []

    for capacity_col, cf_col in pairs:
        if capacity_col in comparison_df.columns and cf_col in comparison_df.columns:
            correlation = comparison_df[capacity_col].corr(comparison_df[cf_col])

            rows.append(
                {
                    "technology": capacity_col,
                    "cf_column": cf_col,
                    "correlation_capacity_vs_cf": correlation,
                }
            )

    if not rows:
        return pd.DataFrame()

    correlation_df = pd.DataFrame(rows).set_index("technology")

    return correlation_df


# =========================================================
# PLOTTING
# =========================================================

def plot_capacity_boxplot(
    capacity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot boxplot of optimized installed capacities across weather years.

    Keeps technologies with zero installed capacity, so unused technologies
    are still shown in the report figure.
    """
    plot_df = capacity_df.copy()

    ordered_columns = plot_df.mean().sort_values(ascending=False).index
    plot_df = plot_df[ordered_columns]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    plot_df.boxplot(
        ax=ax,
        grid=False,
        patch_artist=False,
        showmeans=False,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
            "alpha": 0.8,
        },
    )

    ax.set_ylabel("Installed capacity [MW]", fontsize=REPORT_FONT_SIZE)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=25, labelsize=REPORT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)

def plot_capacity_by_weather_year(
    capacity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot optimized installed capacities by weather year.
    """
    plot_df = capacity_df.loc[:, (capacity_df > 0.0).any(axis=0)].copy()

    ordered_columns = plot_df.mean().sort_values(ascending=False).index
    plot_df = plot_df[ordered_columns]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    plot_df.plot(
        kind="bar",
        ax=ax,
        width=0.78,
    )

    ax.set_ylabel("Installed capacity [MW]", fontsize=REPORT_FONT_SIZE)
    ax.set_xlabel("Weather year", fontsize=REPORT_FONT_SIZE)
    ax.tick_params(axis="x", labelrotation=0, labelsize=REPORT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc="upper right",
        fontsize=REPORT_FONT_SIZE,
        ncol=2,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_cf_boxplots(
    cf_long_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot hourly renewable capacity factor distributions by weather year
    for solar, onshore wind, and offshore wind.
    """
    technologies = ["solar", "onshore wind", "offshore wind"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12.5, 4.8),
        sharey=True,
    )

    for ax, technology in zip(axes, technologies):
        subset = cf_long_df[cf_long_df["technology"] == technology].copy()

        years = sorted(subset["weather_year"].unique())

        data = [
            subset.loc[subset["weather_year"] == year, "capacity_factor"].values
            for year in years
        ]

        ax.boxplot(
            data,
            patch_artist=False,
            showmeans=False,
            medianprops={"color": "black", "linewidth": 1.3},
            boxprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "black", "linewidth": 1.1},
            capprops={"color": "black", "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3,
                "alpha": 0.7,
            },
        )

        ax.set_xlabel("Weather year", fontsize=REPORT_FONT_SIZE)
        ax.set_xticks(range(1, len(years) + 1))
        ax.set_xticklabels(years, rotation=0)
        ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
        ax.grid(axis="y", alpha=0.3)

        ax.text(
            0.04,
            0.94,
            technology,
            transform=ax.transAxes,
            fontsize=REPORT_FONT_SIZE,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
        )

    axes[0].set_ylabel("Capacity factor [-]", fontsize=REPORT_FONT_SIZE)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_cf_vs_capacity_by_year(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot yearly mean renewable CF and installed renewable capacities.
    """
    renewable_pairs = [
        ("solar", "solar_cf_mean", "Solar"),
        ("onshore wind", "onshore_wind_cf_mean", "Onshore wind"),
        ("offshore wind", "offshore_wind_cf_mean", "Offshore wind"),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8.8, 9.5),
        sharex=True,
    )

    for ax, (capacity_col, cf_col, label) in zip(axes, renewable_pairs):
        if capacity_col not in comparison_df.columns or cf_col not in comparison_df.columns:
            continue

        ax_secondary = ax.twinx()

        ax.plot(
            comparison_df.index,
            comparison_df[capacity_col],
            marker="o",
            linewidth=1.8,
            label="Capacity",
        )

        ax_secondary.plot(
            comparison_df.index,
            comparison_df[cf_col],
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color="black",
            label="Mean CF",
        )

        ax.set_ylabel("Capacity [MW]", fontsize=REPORT_FONT_SIZE)
        ax_secondary.set_ylabel("Mean CF [-]", fontsize=REPORT_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=REPORT_FONT_SIZE)
        ax_secondary.tick_params(axis="y", labelsize=REPORT_FONT_SIZE)
        ax.grid(axis="y", alpha=0.3)

        ax.text(
            0.02,
            0.90,
            label,
            transform=ax.transAxes,
            fontsize=REPORT_FONT_SIZE,
            verticalalignment="top",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.8",
                "alpha": 0.9,
            },
        )

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_secondary.get_legend_handles_labels()

        ax.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
            fontsize=REPORT_FONT_SIZE,
            frameon=True,
            framealpha=0.9,
        )

    axes[-1].set_xlabel("Weather year", fontsize=REPORT_FONT_SIZE)
    axes[-1].tick_params(axis="x", labelsize=REPORT_FONT_SIZE)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


# =========================================================
# PRINTING
# =========================================================

def print_analysis_summary(
    sensitivity_df: pd.DataFrame,
    cf_capacity_correlation_df: pd.DataFrame,
) -> None:
    """
    Print a compact console summary to help interpret the results.
    """
    print("\nSensitivity ranking by absolute capacity range:")

    for technology, row in sensitivity_df.iterrows():
        relative_range = row["relative_range"]

        if pd.isna(relative_range):
            relative_range_text = "n/a"
        else:
            relative_range_text = f"{relative_range:.3f}"

        print(
            f"  {technology}: mean={row['mean_capacity_mw']:.1f} MW, "
            f"range={row['range_capacity_mw']:.1f} MW, "
            f"relative_range={relative_range_text}"
        )

    if not cf_capacity_correlation_df.empty:
        print("\nCorrelation between mean CF and installed capacity:")

        for technology, row in cf_capacity_correlation_df.iterrows():
            print(
                f"  {technology}: corr(capacity, CF) = "
                f"{row['correlation_capacity_vs_cf']:.3f}"
            )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Run complete weather sensitivity analysis.
    """
    set_report_plot_style()
    ensure_output_dir(OUTPUT_DIR)

    capacity_df = build_capacity_table(
        network_dir=NETWORK_DIR,
        years=YEARS,
        carrier_aliases=CARRIER_ALIASES,
    )

    cf_df = build_cf_summary(
        timeseries_file=TIMESERIES_FILE,
        country_code=COUNTRY_CODE,
        years=YEARS,
    )

    cf_long_df = build_cf_timeseries_long(
        timeseries_file=TIMESERIES_FILE,
        country_code=COUNTRY_CODE,
        years=YEARS,
    )

    sensitivity_df = compute_capacity_sensitivity_metrics(capacity_df)
    comparison_df = build_comparison_table(capacity_df, cf_df)
    cf_capacity_correlation_df = compute_cf_capacity_correlations(comparison_df)

    print("\nInstalled capacities by weather year [MW]:")
    print(capacity_df.round(2))

    print("\nRenewable mean capacity factors by weather year:")
    print(cf_df.round(4))

    print("\nSensitivity metrics:")
    print(sensitivity_df.round(3))

    print("\nCombined comparison table:")
    print(comparison_df.round(4))

    print("\nWind advantage (offshore - onshore CF):")
    print(cf_df["wind_advantage"].round(4))

    if {"onshore wind", "offshore wind", "wind_advantage"}.issubset(
        comparison_df.columns
    ):
        print("\nWind competition overview:")
        print(
            comparison_df[
                ["onshore wind", "offshore wind", "wind_advantage"]
            ].round(3)
        )

    if not cf_capacity_correlation_df.empty:
        print("\nCF-capacity correlations:")
        print(cf_capacity_correlation_df.round(3))

    print_analysis_summary(
        sensitivity_df=sensitivity_df,
        cf_capacity_correlation_df=cf_capacity_correlation_df,
    )

    capacity_df.to_csv(OUTPUT_DIR / "capacity_by_weather_year.csv")
    cf_df.to_csv(OUTPUT_DIR / "cf_summary_by_weather_year.csv")
    sensitivity_df.to_csv(OUTPUT_DIR / "capacity_sensitivity_metrics.csv")
    comparison_df.to_csv(OUTPUT_DIR / "capacity_cf_comparison.csv")

    if not cf_capacity_correlation_df.empty:
        cf_capacity_correlation_df.to_csv(
            OUTPUT_DIR / "cf_capacity_correlations.csv"
        )

    plot_capacity_boxplot(
        capacity_df=capacity_df,
        output_path=OUTPUT_DIR / "capacity_boxplot_weather_years.png",
    )

    plot_capacity_by_weather_year(
        capacity_df=capacity_df,
        output_path=OUTPUT_DIR / "capacity_by_weather_year.png",
    )

    plot_cf_boxplots(
        cf_long_df=cf_long_df,
        output_path=OUTPUT_DIR / "cf_boxplot_weather_years.png",
    )

    plot_cf_vs_capacity_by_year(
        comparison_df=comparison_df,
        output_path=OUTPUT_DIR / "cf_vs_capacity_by_weather_year.png",
    )

    print("\nWeather sensitivity analysis finished.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()