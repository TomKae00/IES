
from calendar import week
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from interconnectors_analyze import load_network


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

def get_carrier_colors(n: pypsa.Network, columns: pd.Index) -> list[str]:
    """
    Return colors for carriers based on n.carriers['color'].
    """
    fallback_color = "#999999"
    colors = []

    for carrier in columns:
        if carrier in n.carriers.index and pd.notna(n.carriers.at[carrier, "color"]):
            colors.append(n.carriers.at[carrier, "color"])
        else:
            colors.append(fallback_color)

    return colors


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


def plot_annual_mix_from_balance(n: pypsa.Network, output_dir: Path) -> None:

    balance = n.statistics.energy_balance(aggregate_time=False)

    # group by carrier, with time on rows
    balance_by_carrier = balance.groupby(level="carrier").sum().T

    # drop empty/all-zero carriers
    balance_by_carrier = balance_by_carrier.dropna(axis=1, how="all")
    balance_by_carrier = balance_by_carrier.loc[
        :, (balance_by_carrier.fillna(0).abs() > 0).any(axis=0)
    ]

    # remove bookkeeping / unwanted carriers
    balance_by_carrier = balance_by_carrier.drop(
        columns=["electricity", "battery", "battery charger", "battery discharger"],
        errors="ignore",
    )

    # keep only positive contributions
    positive_balance = balance_by_carrier.clip(lower=0)

    # annual generation by carrier
    annual_mix = positive_balance.sum(axis=0)

    # remove any zero entries after clipping
    annual_mix = annual_mix[annual_mix > 0]

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = get_carrier_colors(n, annual_mix.index)

    wedges, texts, autotexts = ax.pie(
        annual_mix,
        colors=colors,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"},
        textprops={"color": "black"},    
    )

    ax.legend(
        wedges,
        annual_mix.index,
        title="Technology",
        loc="lower right",
        bbox_to_anchor=(1, 0),
        fontsize=10,
    )

    ax.set_title("Annual electricity mix")

    fig.tight_layout()
    fig.savefig(output_dir / "annual_electricity_mix_from_balance.png", dpi=300)
    plt.close(fig)


def plot_balance_week2(n: pypsa.Network, output_dir: Path, season: str) -> None:
    """
    Plot the system balance for a representative week, with electricity demand
    shown as a positive line.
    """
    season_weeks = get_representative_weeks(n.snapshots)

    balance = n.statistics.energy_balance(aggregate_time=False)

    # Aggregate by carrier
    balance_by_carrier = balance.groupby(level="carrier").sum()

    # Put time on x-axis
    balance_by_carrier_t = balance_by_carrier.T

    # Drop carriers that are all NaN
    balance_by_carrier_t = balance_by_carrier_t.dropna(axis=1, how="all")

    # Drop carriers that are zero everywhere
    balance_by_carrier_t = balance_by_carrier_t.loc[
        :, (balance_by_carrier_t.fillna(0).abs() > 0).any(axis=0)
    ]

    # Select representative week
    week = balance_by_carrier_t.loc[season_weeks[season]].copy()

    # Electricity demand as positive line
    # Assumes electricity loads are attached to electricity buses
    elec_loads = n.loads.index[n.loads.bus.map(n.buses.carrier) == "electricity"]
    demand = n.loads_t.p_set[elec_loads].sum(axis=1).loc[season_weeks[season]]

    relevant = ["gas", "offwind", "solar"]
    week = week[relevant] if all(col in week.columns for col in relevant) else pd.DataFrame()

    # Colors
    colors = get_carrier_colors(n, week.columns)

    # Styling
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    fig, ax = plt.subplots(figsize=(13, 6))

    # Positive and negative stacked areas
    week.plot.area(ax=ax, stacked=True, color=colors)

    # Demand line
    demand.plot(ax=ax, color="black", linewidth=2, label="electricity demand")

    ax.set_ylabel("Power [MW]")
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    # Put legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="")

    fig.tight_layout()
    fig.savefig(output_dir / f"balance_{season}_week.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_duration_curves(n: pypsa.Network, output_dir: Path) -> None:


    balance = n.statistics.energy_balance(aggregate_time=False)

    # group by carrier and put time on rows
    balance_by_carrier = balance.groupby(level="carrier").sum().T
    balance_by_carrier = balance_by_carrier.drop(columns=["electricity"], errors="ignore")

    # clean up
    balance_by_carrier = balance_by_carrier.dropna(axis=1, how="all")
    balance_by_carrier = balance_by_carrier.loc[
        :, (balance_by_carrier.fillna(0).abs() > 0).any(axis=0)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for carrier in balance_by_carrier.columns:
        series = balance_by_carrier[carrier].dropna()
        duration = series[series > 0].sort_values(ascending=False).reset_index(drop=True)

        color = get_carrier_colors(n, pd.Index([carrier]))[0]

        ax.plot(duration, label=carrier, color=color)

    ax.set_title("Energy balance duration curves by carrier")
    ax.set_xlabel("Hour rank")
    ax.set_ylabel("Power [MW]")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig.tight_layout()
    fig.savefig(output_dir / "dispatch_duration_curves.png", dpi=300)
    plt.close(fig)


def plot_capacity_factors_over_year(n: pypsa.Network, bus_carrier: str, output_dir: Path) -> None:
    # Select generators at the given bus carrier
    gens = n.generators.index[n.generators.bus.map(n.buses.carrier) == bus_carrier]

    # Time series and capacities
    p = n.generators_t.p[gens]
    p_nom = n.generators.loc[gens, "p_nom_opt"]
    weights = n.snapshot_weightings.generators

    # Weighted generation
    generation = p.multiply(weights, axis=0)

    # 🔧 FIX: group by carrier (no axis=1)
    generation = generation.T.groupby(n.generators.loc[gens, "carrier"]).sum().T

    # Aggregate capacities by carrier
    capacities = p_nom.groupby(n.generators.loc[gens, "carrier"]).sum()

    # Aggregate over time (weekly or monthly)
    generation = generation.resample("ME").sum()
    hours = weights.resample("ME").sum()

    # Capacity factor
    cf = generation.divide(hours, axis=0)
    cf = cf.divide(capacities, axis=1)

    relevant = ["gas", "offwind", "solar"]
    cf = cf[relevant]   

    colors = get_carrier_colors(n, cf.columns)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    cf.plot(ax=ax, color=colors)

    ax.set_ylabel("Capacity factor [-]")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="")

    fig.tight_layout()
    fig.savefig(output_dir / "capacity_factors_over_year.png", dpi=300)
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

    dispatch_ts = get_energy_balance_timeseries(n, bus_carrier)

    # Export core tables
    export_summary_tables(n, bus_carrier, output_dir)
    save_dataframe_csv(dispatch_ts, output_dir / "dispatch_timeseries_mw.csv")

    # Built-in PyPSA statistics plots
    plot_annual_mix_from_balance(n, output_dir)
    plot_capacity_factors_over_year(n, bus_carrier, output_dir)
    

    # Seasonal dispatch and duration curves
    plot_balance_week2(n, output_dir, "winter")
    plot_balance_week2(n, output_dir, "summer")
    plot_duration_curves(n, output_dir)



    print("\nTask 1 analysis finished.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()