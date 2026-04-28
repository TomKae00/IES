from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa


NETWORK_DIR = Path("../results")
TIMESERIES_FILE = ("Data/time_series_60min_singleindex_filtered_2015-2020.csv")
OUTPUT_DIR = Path("../results/weather_sensitivity_analysis")

YEARS = [2015, 2016, 2017, 2018, 2019]
COUNTRY_CODE = "DK"

# Map report-friendly technology names to possible carrier names in the PyPSA networks
CARRIER_ALIASES = {
    "solar": ["solar"],
    "onshore wind": ["onwind", "onshore wind", "wind_onshore", "onshore"],
    "offshore wind": ["offwind", "offshore wind", "wind_offshore", "offshore"],
    "CCGT": ["CCGT", "ccgt", "gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
}


def ensure_output_dir(path: Path) -> None:
    """
    Create output directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def load_country_timeseries(
    timeseries_file: str,
    country_code: str,
    year: str,
) -> dict:
    """
    Load electricity demand and renewable generation data for one country and
    compute capacity factors for solar, onshore wind and offshore wind.

    Missing technologies are returned as zero time series.

    Parameters
    ----------
    timeseries_file : str
        Path to the CSV file containing hourly electricity data.
    country_code : str
        Country code used in the column names, e.g. "DK", "DE", "SE", "NO".
    year : str
        Year to extract from the dataset.

    Returns
    -------
    dict
        Dictionary containing electricity demand and renewable capacity
        factor time series for the selected country.
    """
    raw_timeseries = pd.read_csv(timeseries_file)

    raw_timeseries["utc_timestamp"] = pd.to_datetime(
        raw_timeseries["utc_timestamp"], utc=True
    )
    raw_timeseries = raw_timeseries.set_index("utc_timestamp")
    raw_timeseries.index = raw_timeseries.index.tz_localize(None)

    yearly_timeseries = raw_timeseries.loc[f"{year}-01-01":f"{year}-12-31"]

    electricity_load = yearly_timeseries[
        f"{country_code}_load_actual_entsoe_transparency"
    ].copy()

    zero_series = pd.Series(0.0, index=yearly_timeseries.index)

    def capacity_factor_from_columns(
        generation_column: str,
        capacity_column: str,
    ) -> pd.Series:
        if (
            generation_column in yearly_timeseries.columns
            and capacity_column in yearly_timeseries.columns
        ):
            capacity = yearly_timeseries[capacity_column].replace(0, pd.NA)
            return (
                yearly_timeseries[generation_column] / capacity
            ).fillna(0.0).clip(0.0, 1.0)
        return zero_series.copy()

    solar_capacity_factor = capacity_factor_from_columns(
        generation_column=f"{country_code}_solar_generation_actual",
        capacity_column=f"{country_code}_solar_capacity",
    )

    onshore_wind_capacity_factor = capacity_factor_from_columns(
        generation_column=f"{country_code}_wind_onshore_generation_actual",
        capacity_column=f"{country_code}_wind_onshore_capacity",
    )

    offshore_wind_capacity_factor = capacity_factor_from_columns(
        generation_column=f"{country_code}_wind_offshore_generation_actual",
        capacity_column=f"{country_code}_wind_offshore_capacity",
    )

    timeseries_data = {
        "load": electricity_load,
        "solar_cf": solar_capacity_factor,
        "onshore_wind_cf": onshore_wind_capacity_factor,
        "offshore_wind_cf": offshore_wind_capacity_factor,
    }

    return timeseries_data


def extract_generator_capacities(
    network_path: Path,
    carrier_aliases: dict[str, list[str]],
) -> pd.Series:
    """
    Extract optimized installed generator capacities grouped into report-friendly
    technology categories.

    Parameters
    ----------
    network_path : Path
        Path to solved PyPSA network file.
    carrier_aliases : dict[str, list[str]]
        Mapping from report technology names to possible PyPSA carrier names.

    Returns
    -------
    pd.Series
        Optimized capacities in MW for the requested technologies.
    """
    network = pypsa.Network(network_path)
    capacities_by_carrier = network.generators.groupby("carrier")["p_nom_opt"].sum()

    extracted_capacities = {}

    for technology, aliases in carrier_aliases.items():
        matching_aliases = [alias for alias in aliases if alias in capacities_by_carrier.index]
        extracted_capacities[technology] = capacities_by_carrier.loc[matching_aliases].sum()

    return pd.Series(extracted_capacities, dtype=float)


def build_capacity_table(
    network_dir: Path,
    years: list[int],
    carrier_aliases: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Build a table of optimized installed capacities by weather year.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by weather year with technologies as columns.
    """
    rows = []

    for year in years:
        network_path = network_dir / f"dk_base_battery_network_{year}.nc"

        if not network_path.exists():
            raise FileNotFoundError(f"Network file not found: {network_path}")

        capacities = extract_generator_capacities(
            network_path=network_path,
            carrier_aliases=carrier_aliases,
        )
        capacities.name = year
        rows.append(capacities)

    capacity_df = pd.DataFrame(rows)
    capacity_df.index.name = "weather_year"

    return capacity_df


def build_cf_summary(
    timeseries_file: str,
    country_code: str,
    years: list[int],
) -> pd.DataFrame:
    """
    Build yearly summary statistics for renewable capacity factors and load.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by weather year.
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

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        weather_year, technology, capacity_factor
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
            if series.empty:
                continue

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


def compute_capacity_sensitivity_metrics(capacity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple variability metrics for installed capacities across weather years.

    Returns
    -------
    pd.DataFrame
        Summary metrics by technology.
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

    Returns
    -------
    pd.DataFrame
        Combined table for interpretation.
    """
    comparison_df = capacity_df.join(cf_df, how="left")
    return comparison_df


def compute_cf_capacity_correlations(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple correlations between yearly mean CF and installed capacity for
    the renewable technologies.

    Returns
    -------
    pd.DataFrame
        Correlation summary.
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

    correlation_df = pd.DataFrame(rows).set_index("technology")
    return correlation_df


def plot_capacity_boxplot(
    capacity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot boxplot of optimized installed capacities across weather years.
    """
    ordered_columns = capacity_df.mean().sort_values(ascending=False).index
    plot_df = capacity_df[ordered_columns]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    plot_df.boxplot(ax=ax)

    ax.set_ylabel("Installed capacity [MW]", fontsize=14)
    ax.set_xlabel("Generator", fontsize=14)
    ax.tick_params(axis="x", labelrotation=35, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
        figsize=(12.5, 4.6),
        sharey=True,
    )

    for ax, technology in zip(axes, technologies):
        subset = cf_long_df[cf_long_df["technology"] == technology].copy()

        years = sorted(subset["weather_year"].unique())
        data = [
            subset.loc[subset["weather_year"] == year, "capacity_factor"].values
            for year in years
        ]

        ax.boxplot(data)

        ax.set_title(technology.capitalize(), fontsize=13)
        ax.set_xlabel("Weather year", fontsize=12)
        ax.set_xticklabels(years, rotation=0)
        ax.tick_params(axis="both", labelsize=10)

    axes[0].set_ylabel("Capacity factor [-]", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cf_vs_capacity_by_year(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot yearly mean renewable CF and installed renewable capacities.

    This figure is mainly for your own analysis and interpretation. It does not
    necessarily need to go into the report.
    """
    renewable_pairs = [
        ("solar", "solar_cf_mean", "Solar"),
        ("onshore wind", "onshore_wind_cf_mean", "Onshore wind"),
        ("offshore wind", "offshore_wind_cf_mean", "Offshore wind"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.8, 9.5), sharex=True)

    for ax, (capacity_col, cf_col, title) in zip(axes, renewable_pairs):
        if capacity_col not in comparison_df.columns or cf_col not in comparison_df.columns:
            continue

        ax_secondary = ax.twinx()

        ax.plot(
            comparison_df.index,
            comparison_df[capacity_col],
            marker="o",
            linewidth=1.8,
        )
        ax_secondary.plot(
            comparison_df.index,
            comparison_df[cf_col],
            marker="s",
            linestyle="--",
            linewidth=1.8,
        )

        ax.set_ylabel("Capacity [MW]", fontsize=11)
        ax_secondary.set_ylabel("Mean CF [-]", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax_secondary.tick_params(axis="y", labelsize=10)

    axes[-1].set_xlabel("Weather year", fontsize=12)
    axes[-1].tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_analysis_summary(
    sensitivity_df: pd.DataFrame,
    cf_capacity_correlation_df: pd.DataFrame,
) -> None:
    """
    Print a compact console summary to help interpret the results.
    """
    print("\nSensitivity ranking by absolute capacity range:")
    for technology, row in sensitivity_df.iterrows():
        print(
            f"  {technology}: mean={row['mean_capacity_mw']:.1f} MW, "
            f"range={row['range_capacity_mw']:.1f} MW, "
            f"relative_range={row['relative_range']:.3f}"
        )

    if not cf_capacity_correlation_df.empty:
        print("\nCorrelation between mean CF and installed capacity:")
        for technology, row in cf_capacity_correlation_df.iterrows():
            print(
                f"  {technology}: corr(capacity, CF) = "
                f"{row['correlation_capacity_vs_cf']:.3f}"
            )


def main() -> None:
    """
    Run complete weather sensitivity analysis.
    """
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

    plot_cf_boxplots(
        cf_long_df=cf_long_df,
        output_path=OUTPUT_DIR / "cf_boxplot_weather_years.png",
    )

    plot_cf_vs_capacity_by_year(
        comparison_df=comparison_df,
        output_path=OUTPUT_DIR / "cf_vs_capacity_by_weather_year.png",
    )

if __name__ == "__main__":
    main()