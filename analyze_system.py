from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa


# =========================================================
# CONFIG
# =========================================================

NETWORK_FILE = "results/dk_network_2016.nc"
OUTPUT_DIR = "results/task1_analysis"

# =========================================================
# HELPERS
# =========================================================

def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_single_bus_carrier(n: pypsa.Network) -> str:
    """
    Return the unique bus carrier of the network.
    This is appropriate for a one-node / one-sector electricity model.
    """
    carriers = pd.Index(n.buses.carrier.dropna().unique())

    if len(carriers) != 1:
        raise ValueError(
            f"Expected exactly one bus carrier, found {list(carriers)}. "
            "Set the bus carrier consistently in the model first."
        )

    return carriers[0]


def get_representative_weeks(snapshots: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    """
    Select one full winter week and one full summer week from the snapshots
    without hard-coding a specific year.

    Strategy:
    - winter months: Dec, Jan, Feb
    - summer months: Jun, Jul, Aug
    - take the first full 7-day block found in each season
    """
    if not isinstance(snapshots, pd.DatetimeIndex):
        raise TypeError("Snapshots must be a pandas.DatetimeIndex.")

    def first_full_week(months: list[int]) -> pd.DatetimeIndex:
        season_snaps = snapshots[snapshots.month.isin(months)]
        if len(season_snaps) == 0:
            raise ValueError(f"No snapshots found for months {months}.")

        # Find first timestamp that allows 7 full days inside the same month subset
        for start in season_snaps:
            end = start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
            week = snapshots[(snapshots >= start) & (snapshots <= end)]
            if len(week) == 24 * 7:
                return week

        raise ValueError(f"Could not find a full 7-day period for months {months}.")

    return {
        "winter": first_full_week([12, 1, 2]),
        "summer": first_full_week([6, 7, 8]),
    }


def save_series_csv(series: pd.Series, filepath: Path, index_name: str = "carrier") -> None:
    df = series.rename(series.name if series.name else "value").to_frame()
    df.index.name = index_name
    df.to_csv(filepath)


def save_dataframe_csv(df: pd.DataFrame, filepath: Path) -> None:
    df.to_csv(filepath)


# =========================================================
# STATISTICS EXTRACTION
# =========================================================

def get_optimal_capacity(n: pypsa.Network) -> pd.Series:
    """
    Optimal capacities by carrier using PyPSA statistics.
    """
    s = n.statistics.optimal_capacity(
        comps=["Generator"],
        aggregate_groups="sum",
        nice_names=False,
    )

    if isinstance(s, pd.DataFrame):
        s = s.squeeze()

    s.name = "optimal_capacity"
    return s.sort_values(ascending=False)


def get_annual_mix(n: pypsa.Network, bus_carrier: str) -> pd.Series:
    """
    Annual electricity mix from PyPSA statistics energy balance.
    Positive values correspond to supply contributions.
    """
    s = n.statistics.energy_balance(
        comps=["Generator"],
        bus_carrier=bus_carrier,
        aggregate_time="sum",
        aggregate_groups="sum",
        nice_names=False,
    )

    if isinstance(s, pd.DataFrame):
        s = s.squeeze()

    s = s[s > 0]
    s.name = "annual_generation"
    return s.sort_values(ascending=False)


def get_capacity_factor(n: pypsa.Network, bus_carrier: str) -> pd.Series:
    """
    Capacity factors by carrier from PyPSA statistics.
    """
    s = n.statistics.capacity_factor(
        comps=["Generator"],
        bus_carrier=bus_carrier,
        aggregate_groups="sum",
        nice_names=False,
    )

    if isinstance(s, pd.DataFrame):
        s = s.squeeze()

    s.name = "capacity_factor"
    return s.sort_values(ascending=False)


def get_curtailment(n: pypsa.Network, bus_carrier: str) -> pd.Series:
    """
    Curtailment by carrier from PyPSA statistics.
    """
    s = n.statistics.curtailment(
        comps=["Generator"],
        bus_carrier=bus_carrier,
        aggregate_time="sum",
        aggregate_groups="sum",
        nice_names=False,
    )

    if isinstance(s, pd.DataFrame):
        s = s.squeeze()

    s.name = "curtailment"
    return s.sort_values(ascending=False)


def get_energy_balance_timeseries(n: pypsa.Network, bus_carrier: str) -> pd.DataFrame:
    """
    Time series energy balance in MW from PyPSA statistics.

    aggregate_time=False returns time-dependent values, which is what we want
    for weekly dispatch plots. PyPSA documents this specifically for temporal
    analysis and stacked area style plots. :contentReference[oaicite:1]{index=1}
    """
    df = n.statistics.energy_balance(
        comps=["Generator"],
        bus_carrier=bus_carrier,
        aggregate_time=False,
        aggregate_groups="sum",
        nice_names=False,
    )

    # Make sure snapshots are on rows for plotting convenience
    if df.columns.equals(n.snapshots):
        df = df.T

    # Keep only positive supply side for generation dispatch plots
    df = df.clip(lower=0)

    return df


# =========================================================
# PLOTTING (MANUAL - PANDAS / MATPLOTLIB)
# =========================================================

def plot_optimal_capacities(n: pypsa.Network, output_dir: Path) -> None:
    s = get_optimal_capacity(n)

    fig, ax = plt.subplots(figsize=(8, 4))
    s.plot(kind="bar", ax=ax)

    ax.set_title("Optimal generator capacities")
    ax.set_ylabel("Capacity [MW]")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "optimal_capacities.png", dpi=300)
    plt.close(fig)


def plot_annual_mix(n: pypsa.Network, bus_carrier: str, output_dir: Path) -> None:
    s = get_annual_mix(n, bus_carrier)

    fig, ax = plt.subplots(figsize=(8, 4))
    s.plot(kind="bar", ax=ax)

    ax.set_title("Annual electricity mix")
    ax.set_ylabel("Generation [MWh]")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "annual_electricity_mix.png", dpi=300)
    plt.close(fig)


def plot_capacity_factors(n: pypsa.Network, bus_carrier: str, output_dir: Path) -> None:
    s = get_capacity_factor(n, bus_carrier)

    fig, ax = plt.subplots(figsize=(8, 4))
    s.plot(kind="bar", ax=ax)

    ax.set_title("Capacity factors by technology")
    ax.set_ylabel("Capacity factor [-]")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "capacity_factors.png", dpi=300)
    plt.close(fig)


def plot_curtailment(n: pypsa.Network, bus_carrier: str, output_dir: Path) -> None:
    s = get_curtailment(n, bus_carrier)

    fig, ax = plt.subplots(figsize=(8, 4))
    s.plot(kind="bar", ax=ax)

    ax.set_title("Curtailment by technology")
    ax.set_ylabel("Curtailment [MWh]")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "curtailment.png", dpi=300)
    plt.close(fig)


def plot_dispatch_week(
    dispatch_ts: pd.DataFrame,
    week_snapshots: pd.DatetimeIndex,
    season: str,
    output_dir: Path,
) -> None:
    week = dispatch_ts.loc[week_snapshots]

    fig, ax = plt.subplots(figsize=(12, 4))
    week.plot.area(ax=ax, linewidth=0)

    ax.set_title(f"Dispatch time series - {season.capitalize()} week")
    ax.set_ylabel("Dispatch [MW]")
    ax.set_xlabel("")
    ax.legend(title="Technology", ncol=3, fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"dispatch_{season}_week.png", dpi=300)
    plt.close(fig)


def plot_duration_curves(dispatch_ts: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for carrier in dispatch_ts.columns:
        duration = (
            dispatch_ts[carrier]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        ax.plot(duration, label=carrier)

    ax.set_title("Dispatch duration curves")
    ax.set_xlabel("Hour rank")
    ax.set_ylabel("Dispatch [MW]")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "dispatch_duration_curves.png", dpi=300)
    plt.close(fig)


# =========================================================
# SUMMARY TABLES
# =========================================================

def export_summary_tables(n: pypsa.Network, bus_carrier: str, output_dir: Path) -> None:
    optimal_capacity = get_optimal_capacity(n)
    annual_mix = get_annual_mix(n, bus_carrier)
    capacity_factor = get_capacity_factor(n, bus_carrier)
    curtailment = get_curtailment(n, bus_carrier)

    optimal_capacity = optimal_capacity.groupby("carrier").sum()
    annual_mix = annual_mix.groupby("carrier").sum()
    capacity_factor = capacity_factor.groupby("carrier").mean()
    curtailment = curtailment.groupby("carrier").sum()

    summary = pd.concat(
        [optimal_capacity, annual_mix, capacity_factor, curtailment],
        axis=1,
    )

    summary.columns = [
        "optimal_capacity_mw",
        "annual_generation_mwh",
        "capacity_factor",
        "curtailment_mwh",
    ]

    summary = summary.sort_values("annual_generation_mwh", ascending=False)
    summary.to_csv(output_dir / "task1_summary.csv")

    save_series_csv(optimal_capacity, output_dir / "optimal_capacity.csv")
    save_series_csv(annual_mix, output_dir / "annual_mix.csv")
    save_series_csv(capacity_factor, output_dir / "capacity_factor.csv")
    save_series_csv(curtailment, output_dir / "curtailment.csv")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    output_dir = ensure_output_dir(OUTPUT_DIR)
    n = pypsa.Network(NETWORK_FILE)
    n.sanitize()


    bus_carrier = get_single_bus_carrier(n)
    print(f"Detected bus carrier: {bus_carrier}")

    season_weeks = get_representative_weeks(n.snapshots)
    dispatch_ts = get_energy_balance_timeseries(n, bus_carrier)

    # Export core tables
    export_summary_tables(n, bus_carrier, output_dir)
    save_dataframe_csv(dispatch_ts, output_dir / "dispatch_timeseries_mw.csv")

    # Built-in PyPSA statistics plots
    plot_optimal_capacities(n, output_dir)
    plot_annual_mix(n, bus_carrier, output_dir)
    plot_capacity_factors(n, bus_carrier, output_dir)
    plot_curtailment(n, bus_carrier, output_dir)

    # Seasonal dispatch and duration curves
    plot_dispatch_week(dispatch_ts, season_weeks["winter"], "winter", output_dir)
    plot_dispatch_week(dispatch_ts, season_weeks["summer"], "summer", output_dir)
    plot_duration_curves(dispatch_ts, output_dir)

    print("\nTask 1 analysis finished.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()