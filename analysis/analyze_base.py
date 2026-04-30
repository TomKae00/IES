from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETWORK_DIR = PROJECT_ROOT / "results" / "networks"
OUTPUT_DIR = PROJECT_ROOT / "results" / "task1_analysis"

NETWORK_NAME = "base_DK_2016.nc"

CARRIER_ALIASES = {
    "solar": ["solar"],
    "onshore wind": ["onwind", "onshore wind", "wind_onshore", "onshore"],
    "offshore wind": ["offwind", "offshore wind", "wind_offshore", "offshore"],
    "CCGT": ["gas CCGT", "CCGT", "ccgt", "gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
    "electricity": ["electricity", "AC"],
    "battery": ["battery"],
    "battery charger": ["battery charger"],
    "battery discharger": ["battery discharger"],
}

DISPLAY_NAME_TO_CARRIER = {
    "solar": "solar",
    "onshore wind": "onwind",
    "offshore wind": "offwind",
    "CCGT": "gas CCGT",
    "coal": "coal",
    "nuclear": "nuclear",
    "electricity": "AC",
    "AC": "AC",
    "battery": "battery",
    "battery charger": "battery charger",
    "battery discharger": "battery discharger",
}

DEFAULT_CARRIER_COLORS = {
    # Electricity
    "AC": "#4C566A",
    "electricity": "#4C566A",

    # Electricity generation
    "solar": "#EBCB3B",
    "onwind": "#5AA469",
    "onshore wind": "#5AA469",
    "offwind": "#2E86AB",
    "offshore wind": "#2E86AB",
    "gas CCGT": "#D08770",
    "CCGT": "#D08770",
    "coal": "#5C5C5C",
    "nuclear": "#8F6BB3",

    # Battery
    "battery": "#E67E22",
    "battery charger": "#C06C84",
    "battery discharger": "#6C5B7B",
}

# General positive carrier order for summary plots, duration curves, etc.
POSITIVE_CARRIER_ORDER = [
    "solar",
    "offshore wind",
    "onshore wind",
    "CCGT",
    "coal",
    "nuclear",
    "battery discharger",
]

# Energy balance stacking order.
# Negative demand stays below zero, while positive generation is stacked as:
# solar at the bottom, then offshore wind, then onshore wind, then CCGT.
ENERGY_BALANCE_CARRIER_ORDER = [
    "electricity",
    "battery charger",
    "solar",
    "offshore wind",
    "onshore wind",
    "CCGT",
    "coal",
    "nuclear",
    "battery discharger",
]

REPORT_FONT_SIZE = 14


# =========================================================
# PLOT STYLE
# =========================================================

def set_report_plot_style() -> None:
    """
    Set global matplotlib style for report-ready plots.
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

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")


# =========================================================
# HELPERS
# =========================================================

def ensure_output_dir(path: Path) -> None:
    """
    Create output directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def get_network_path(network_dir: Path) -> Path:
    """
    Return the expected solved network path.
    """
    return network_dir / NETWORK_NAME


def load_network(network_path: Path) -> pypsa.Network:
    """
    Load solved PyPSA network from NetCDF.
    """
    if not network_path.exists():
        raise FileNotFoundError(
            f"Network file not found: {network_path}\n"
            "Make sure the solved network exists in results/networks."
        )

    network = pypsa.Network(network_path)
    network.sanitize()

    return network


def get_single_bus_carrier(network: pypsa.Network) -> str:
    """
    Return the unique bus carrier of the network.

    This is appropriate for the one-node / one-sector electricity model.
    """
    carriers = pd.Index(network.buses.carrier.dropna().unique())

    if len(carriers) != 1:
        raise ValueError(
            f"Expected exactly one bus carrier, found {list(carriers)}. "
            "For this base analysis, the network should only contain one bus carrier."
        )

    return carriers[0]


def map_carrier_to_display_name(carrier: str) -> str:
    """
    Map internal PyPSA carrier names to report-friendly display names.
    """
    for display_name, aliases in CARRIER_ALIASES.items():
        if carrier in aliases:
            return display_name

    return carrier


def get_color_lookup_candidates(carrier: str) -> list[str]:
    """
    Return possible carrier names that can be used for color lookup.
    """
    candidates = []

    if carrier in DISPLAY_NAME_TO_CARRIER:
        candidates.append(DISPLAY_NAME_TO_CARRIER[carrier])

    candidates.append(carrier)

    if carrier in CARRIER_ALIASES:
        candidates.extend(CARRIER_ALIASES[carrier])

    for display_name, aliases in CARRIER_ALIASES.items():
        if carrier in aliases:
            candidates.append(display_name)
            candidates.extend(aliases)

    clean_candidates = []
    for candidate in candidates:
        if candidate not in clean_candidates:
            clean_candidates.append(candidate)

    return clean_candidates


def get_carrier_colors(network: pypsa.Network, carriers: pd.Index) -> list[str]:
    """
    Return colors for plotted carrier names.

    The function first checks colors stored in network.carriers.
    If these are unavailable, it falls back to DEFAULT_CARRIER_COLORS.
    """
    fallback_color = "#999999"
    colors = []

    for carrier in carriers:
        color = None

        for lookup_carrier in get_color_lookup_candidates(carrier):
            if (
                lookup_carrier in network.carriers.index
                and "color" in network.carriers.columns
                and pd.notna(network.carriers.at[lookup_carrier, "color"])
                and network.carriers.at[lookup_carrier, "color"] != ""
            ):
                color = network.carriers.at[lookup_carrier, "color"]
                break

            if lookup_carrier in DEFAULT_CARRIER_COLORS:
                color = DEFAULT_CARRIER_COLORS[lookup_carrier]
                break

        colors.append(color if color is not None else fallback_color)

    return colors


def print_carrier_color_check(network: pypsa.Network, carriers: pd.Index) -> None:
    """
    Print carrier-color mapping to the console for debugging.
    """
    print("\nCarrier color check:")

    for carrier in carriers:
        color = get_carrier_colors(network, pd.Index([carrier]))[0]
        print(f"  {carrier}: {color}")


def drop_empty_carriers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop carriers that are entirely NaN or zero.
    """
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, (df.fillna(0.0).abs() > 0.0).any(axis=0)]

    return df


def reorder_carriers(df: pd.DataFrame, carrier_order: list[str]) -> pd.DataFrame:
    """
    Reorder columns according to a preferred carrier order while keeping
    remaining carriers at the end.
    """
    ordered = [carrier for carrier in carrier_order if carrier in df.columns]
    remaining = [carrier for carrier in df.columns if carrier not in ordered]

    return df[ordered + remaining]


def get_representative_weeks(
    snapshots: pd.DatetimeIndex,
) -> dict[str, pd.DatetimeIndex]:
    """
    Select one full winter week and one full summer week from the snapshots.

    Strategy:
    - winter months: December, January, February
    - summer months: June, July, August
    - take the first full 7-day block found in each season
    """
    if not isinstance(snapshots, pd.DatetimeIndex):
        raise TypeError("Snapshots must be a pandas.DatetimeIndex.")

    def first_full_week(months: list[int]) -> pd.DatetimeIndex:
        season_snapshots = snapshots[snapshots.month.isin(months)]

        if len(season_snapshots) == 0:
            raise ValueError(f"No snapshots found for months {months}.")

        for start in season_snapshots:
            end = start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
            week = snapshots[(snapshots >= start) & (snapshots <= end)]

            if len(week) == 24 * 7:
                return week

        raise ValueError(f"Could not find a full 7-day period for months {months}.")

    return {
        "winter": first_full_week([12, 1, 2]),
        "summer": first_full_week([6, 7, 8]),
    }


def save_series_csv(
    series: pd.Series,
    filepath: Path,
    index_name: str = "carrier",
) -> None:
    """
    Save Series as CSV.
    """
    df = series.rename(series.name if series.name else "value").to_frame()
    df.index.name = index_name
    df.to_csv(filepath)


def save_dataframe_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save DataFrame as CSV.
    """
    df.to_csv(filepath)


# =========================================================
# DATA EXTRACTION
# =========================================================

def get_generators_at_bus_carrier(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.Index:
    """
    Return generators connected to buses of the selected bus carrier.
    """
    generator_bus_carriers = network.generators.bus.map(network.buses.carrier)

    return network.generators.index[generator_bus_carriers == bus_carrier]


def get_load_timeseries(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.Series:
    """
    Return total load time series for the selected bus carrier.
    """
    load_bus_carriers = network.loads.bus.map(network.buses.carrier)
    loads = network.loads.index[load_bus_carriers == bus_carrier]

    if len(loads) == 0:
        return pd.Series(0.0, index=network.snapshots, name="demand")

    demand = network.loads_t.p_set[loads].sum(axis=1)
    demand.name = "demand"

    return demand


def get_generator_dispatch_timeseries(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.DataFrame:
    """
    Return generator dispatch time series grouped by report-friendly carrier name.
    """
    generators = get_generators_at_bus_carrier(
        network=network,
        bus_carrier=bus_carrier,
    )

    if len(generators) == 0:
        raise ValueError(f"No generators found for bus carrier '{bus_carrier}'.")

    dispatch = network.generators_t.p[generators].copy()

    carrier_display_names = network.generators.loc[generators, "carrier"].map(
        map_carrier_to_display_name
    )

    dispatch_by_carrier = dispatch.T.groupby(carrier_display_names).sum().T
    dispatch_by_carrier = dispatch_by_carrier.clip(lower=0.0)

    dispatch_by_carrier = drop_empty_carriers(dispatch_by_carrier)
    dispatch_by_carrier = reorder_carriers(dispatch_by_carrier, POSITIVE_CARRIER_ORDER)

    return dispatch_by_carrier


def get_energy_balance_by_carrier(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.DataFrame:
    """
    Return the time-resolved energy balance at the selected electricity bus carrier,
    aggregated by carrier.

    This uses PyPSA's n.statistics.energy_balance(aggregate_time=False).
    """
    balance = network.statistics.energy_balance(
        aggregate_time=False,
        nice_names=False,
    )

    if "bus_carrier" not in balance.index.names:
        raise KeyError(
            "The energy balance index does not contain a 'bus_carrier' level. "
            f"Available levels are: {balance.index.names}"
        )

    balance_selected = balance.xs(bus_carrier, level="bus_carrier")
    balance_by_carrier = balance_selected.groupby(level="carrier").sum()

    balance_by_carrier_t = balance_by_carrier.T

    if len(balance_by_carrier_t.index) == len(network.snapshots):
        balance_by_carrier_t.index = network.snapshots

    balance_by_carrier_t = balance_by_carrier_t.rename(
        columns=lambda carrier: map_carrier_to_display_name(carrier)
    )

    # If several internal carrier names map to the same display name, sum them.
    balance_by_carrier_t = balance_by_carrier_t.T.groupby(level=0).sum().T

    balance_by_carrier_t = drop_empty_carriers(balance_by_carrier_t)
    balance_by_carrier_t = reorder_carriers(
        balance_by_carrier_t,
        ENERGY_BALANCE_CARRIER_ORDER,
    )

    return balance_by_carrier_t


def get_optimal_capacity(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.Series:
    """
    Return optimized installed generator capacities grouped by carrier.
    """
    generators = get_generators_at_bus_carrier(
        network=network,
        bus_carrier=bus_carrier,
    )

    if len(generators) == 0:
        return pd.Series(dtype=float, name="optimal_capacity_mw")

    if "p_nom_opt" not in network.generators.columns:
        raise KeyError(
            "'p_nom_opt' not found in network.generators. "
            "Make sure the network was optimized before saving."
        )

    capacities = network.generators.loc[generators, "p_nom_opt"].copy()

    carrier_display_names = network.generators.loc[generators, "carrier"].map(
        map_carrier_to_display_name
    )

    capacity_by_carrier = capacities.groupby(carrier_display_names).sum()
    capacity_by_carrier.name = "optimal_capacity_mw"

    return capacity_by_carrier.sort_values(ascending=False)


def get_annual_generation(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.Series:
    """
    Return annual generation in MWh grouped by carrier.
    """
    dispatch = get_generator_dispatch_timeseries(
        network=network,
        bus_carrier=bus_carrier,
    )

    weights = network.snapshot_weightings.generators.reindex(dispatch.index)
    generation = dispatch.multiply(weights, axis=0).sum(axis=0)
    generation = generation[generation > 0.0]

    generation.name = "annual_generation_mwh"

    return generation.sort_values(ascending=False)


def get_capacity_factor(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.Series:
    """
    Return annual capacity factors grouped by carrier.

    Capacity factor is calculated as:
    annual generation / (installed capacity * weighted hours)
    """
    annual_generation = get_annual_generation(
        network=network,
        bus_carrier=bus_carrier,
    )

    optimal_capacity = get_optimal_capacity(
        network=network,
        bus_carrier=bus_carrier,
    )

    weights = network.snapshot_weightings.generators
    weighted_hours = weights.sum()

    denominator = optimal_capacity * weighted_hours
    denominator = denominator.replace(0.0, pd.NA)

    capacity_factor = annual_generation / denominator
    capacity_factor = capacity_factor.dropna()
    capacity_factor.name = "capacity_factor"

    return capacity_factor.sort_values(ascending=False)


def get_curtailment(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.Series:
    """
    Estimate renewable curtailment in MWh.

    Curtailment is only calculated for variable renewable generators where
    p_max_pu limits availability. Thermal generators are assigned zero.
    """
    generators = get_generators_at_bus_carrier(
        network=network,
        bus_carrier=bus_carrier,
    )

    if len(generators) == 0:
        return pd.Series(dtype=float, name="curtailment_mwh")

    weights = network.snapshot_weightings.generators

    p = network.generators_t.p[generators]
    p_nom_opt = network.generators.loc[generators, "p_nom_opt"]

    if network.generators_t.p_max_pu.empty:
        return pd.Series(0.0, index=[], name="curtailment_mwh")

    p_max_pu = network.generators_t.p_max_pu.reindex(
        index=network.snapshots,
        columns=generators,
        fill_value=1.0,
    )

    available = p_max_pu.multiply(p_nom_opt, axis=1)
    curtailed = (available - p).clip(lower=0.0)
    curtailed = curtailed.multiply(weights, axis=0)

    carrier_display_names = network.generators.loc[generators, "carrier"].map(
        map_carrier_to_display_name
    )

    curtailment_by_carrier = curtailed.T.groupby(carrier_display_names).sum().T.sum()

    renewable_carriers = ["solar", "onshore wind", "offshore wind"]
    curtailment_by_carrier = curtailment_by_carrier.reindex(
        renewable_carriers,
        fill_value=0.0,
    )

    curtailment_by_carrier.name = "curtailment_mwh"

    return curtailment_by_carrier.sort_values(ascending=False)


# =========================================================
# SUMMARY TABLES
# =========================================================

def build_summary_table(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.DataFrame:
    """
    Build core summary table for the base network.
    """
    optimal_capacity = get_optimal_capacity(
        network=network,
        bus_carrier=bus_carrier,
    )

    annual_generation = get_annual_generation(
        network=network,
        bus_carrier=bus_carrier,
    )

    capacity_factor = get_capacity_factor(
        network=network,
        bus_carrier=bus_carrier,
    )

    curtailment = get_curtailment(
        network=network,
        bus_carrier=bus_carrier,
    )

    summary = pd.concat(
        [
            optimal_capacity,
            annual_generation,
            capacity_factor,
            curtailment,
        ],
        axis=1,
    )

    summary = summary.fillna(
        {
            "annual_generation_mwh": 0.0,
            "capacity_factor": 0.0,
            "curtailment_mwh": 0.0,
        }
    )

    if "annual_generation_mwh" in summary.columns:
        summary = summary.sort_values("annual_generation_mwh", ascending=False)

    return summary


def export_summary_tables(
    network: pypsa.Network,
    bus_carrier: str,
    output_dir: Path,
) -> None:
    """
    Export all summary tables as CSV files.
    """
    optimal_capacity = get_optimal_capacity(
        network=network,
        bus_carrier=bus_carrier,
    )

    annual_generation = get_annual_generation(
        network=network,
        bus_carrier=bus_carrier,
    )

    capacity_factor = get_capacity_factor(
        network=network,
        bus_carrier=bus_carrier,
    )

    curtailment = get_curtailment(
        network=network,
        bus_carrier=bus_carrier,
    )

    summary = build_summary_table(
        network=network,
        bus_carrier=bus_carrier,
    )

    summary.to_csv(output_dir / "task1_summary.csv")

    save_series_csv(
        optimal_capacity,
        output_dir / "optimal_capacity.csv",
    )

    save_series_csv(
        annual_generation,
        output_dir / "annual_generation.csv",
    )

    save_series_csv(
        capacity_factor,
        output_dir / "capacity_factor.csv",
    )

    save_series_csv(
        curtailment,
        output_dir / "curtailment.csv",
    )


# =========================================================
# PLOTTING
# =========================================================

def plot_annual_generation_mix(
    network: pypsa.Network,
    bus_carrier: str,
    output_path: Path,
) -> None:
    """
    Plot annual electricity generation mix as a pie chart.
    """
    annual_generation = get_annual_generation(
        network=network,
        bus_carrier=bus_carrier,
    )

    annual_generation = annual_generation[annual_generation > 0.0]
    annual_generation = annual_generation[
        reorder_carriers(annual_generation.to_frame().T, POSITIVE_CARRIER_ORDER).columns
    ]

    colors = get_carrier_colors(
        network=network,
        carriers=annual_generation.index,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    wedges, _, _ = ax.pie(
        annual_generation,
        colors=colors,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        textprops={"color": "black", "fontsize": REPORT_FONT_SIZE},
    )

    ax.legend(
        wedges,
        annual_generation.index,
        loc="center right",
        bbox_to_anchor=(0.98, 0.4),
        fontsize=REPORT_FONT_SIZE,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_capacity_bar(
    network: pypsa.Network,
    bus_carrier: str,
    output_path: Path,
) -> None:
    """
    Plot optimized installed capacities by carrier.
    """
    capacities = get_optimal_capacity(
        network=network,
        bus_carrier=bus_carrier,
    )

    capacities = capacities[capacities > 0.0]
    capacities = capacities[
        reorder_carriers(capacities.to_frame().T, POSITIVE_CARRIER_ORDER).columns
    ]

    colors = get_carrier_colors(
        network=network,
        carriers=capacities.index,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    capacities.plot(
        kind="bar",
        ax=ax,
        color=colors,
        width=0.75,
    )

    ax.set_ylabel("Installed capacity [MW]", fontsize=REPORT_FONT_SIZE)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=25, labelsize=REPORT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_capacity_factors_over_year(
    network: pypsa.Network,
    bus_carrier: str,
    output_path: Path,
) -> None:
    """
    Plot monthly capacity factors by carrier.
    """
    generators = get_generators_at_bus_carrier(
        network=network,
        bus_carrier=bus_carrier,
    )

    if len(generators) == 0:
        raise ValueError(f"No generators found for bus carrier '{bus_carrier}'.")

    dispatch = network.generators_t.p[generators].copy()
    weights = network.snapshot_weightings.generators.reindex(dispatch.index)
    capacities = network.generators.loc[generators, "p_nom_opt"].copy()

    carrier_display_names = network.generators.loc[generators, "carrier"].map(
        map_carrier_to_display_name
    )

    weighted_generation = dispatch.multiply(weights, axis=0)
    generation_by_carrier = weighted_generation.T.groupby(carrier_display_names).sum().T
    capacity_by_carrier = capacities.groupby(carrier_display_names).sum()

    monthly_generation = generation_by_carrier.resample("ME").sum()
    monthly_hours = weights.resample("ME").sum()

    monthly_capacity_factor = monthly_generation.divide(monthly_hours, axis=0)
    monthly_capacity_factor = monthly_capacity_factor.divide(capacity_by_carrier, axis=1)

    monthly_capacity_factor = monthly_capacity_factor.replace(
        [float("inf"), -float("inf")],
        pd.NA,
    )
    monthly_capacity_factor = monthly_capacity_factor.dropna(axis=1, how="all")
    monthly_capacity_factor = drop_empty_carriers(monthly_capacity_factor)
    monthly_capacity_factor = reorder_carriers(
        monthly_capacity_factor,
        POSITIVE_CARRIER_ORDER,
    )

    colors = get_carrier_colors(
        network=network,
        carriers=monthly_capacity_factor.columns,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    monthly_capacity_factor.plot(
        ax=ax,
        color=colors,
        linewidth=2.0,
    )

    ax.set_ylabel("Capacity factor [-]", fontsize=REPORT_FONT_SIZE)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc="upper right",
        fontsize=REPORT_FONT_SIZE,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_energy_balance_week(
    network: pypsa.Network,
    balance_by_carrier_t: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_path: Path,
) -> None:
    """
    Plot one representative week of the system energy balance using
    n.statistics.energy_balance().
    """
    week_balance = balance_by_carrier_t.loc[start_date:end_date].copy()
    week_balance = drop_empty_carriers(week_balance)
    week_balance = reorder_carriers(
        week_balance,
        ENERGY_BALANCE_CARRIER_ORDER,
    )

    colors = get_carrier_colors(
        network=network,
        carriers=week_balance.columns,
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    week_balance.plot.area(
        stacked=True,
        ax=ax,
        color=colors,
        linewidth=0.0,
    )

    ax.set_ylabel("Power balance [MW]", fontsize=REPORT_FONT_SIZE)
    ax.set_xlabel("Time", fontsize=REPORT_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc="lower right",
        fontsize=REPORT_FONT_SIZE,
        ncol=2,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_duration_curves(
    network: pypsa.Network,
    bus_carrier: str,
    output_path: Path,
) -> None:
    """
    Plot dispatch duration curves by carrier.
    """
    dispatch = get_generator_dispatch_timeseries(
        network=network,
        bus_carrier=bus_carrier,
    )

    dispatch = reorder_carriers(dispatch, POSITIVE_CARRIER_ORDER)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for carrier in dispatch.columns:
        duration = (
            dispatch[carrier]
            .dropna()
            .clip(lower=0.0)
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )

        duration = duration[duration > 0.0]

        if duration.empty:
            continue

        color = get_carrier_colors(
            network=network,
            carriers=pd.Index([carrier]),
        )[0]

        ax.plot(
            duration,
            label=carrier,
            color=color,
            linewidth=2.0,
        )

    ax.set_xlabel("Hour rank", fontsize=REPORT_FONT_SIZE)
    ax.set_ylabel("Power [MW]", fontsize=REPORT_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.grid(alpha=0.3)

    ax.legend(
        loc="upper right",
        fontsize=REPORT_FONT_SIZE,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


# =========================================================
# PRINTING
# =========================================================

def print_analysis_summary(summary: pd.DataFrame) -> None:
    """
    Print compact summary to console.
    """
    print("\nTask 1 summary:")
    print(summary.round(3))


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Run complete base network analysis.
    """
    set_report_plot_style()
    ensure_output_dir(OUTPUT_DIR)

    network_path = get_network_path(NETWORK_DIR)
    network = load_network(network_path)

    bus_carrier = get_single_bus_carrier(network)

    print(f"Detected bus carrier: {bus_carrier}")
    print(f"Loaded network: {network_path}")

    dispatch = get_generator_dispatch_timeseries(
        network=network,
        bus_carrier=bus_carrier,
    )

    energy_balance_by_carrier = get_energy_balance_by_carrier(
        network=network,
        bus_carrier=bus_carrier,
    )

    summary = build_summary_table(
        network=network,
        bus_carrier=bus_carrier,
    )

    print_analysis_summary(summary)

    print_carrier_color_check(
        network=network,
        carriers=pd.Index(energy_balance_by_carrier.columns),
    )

    export_summary_tables(
        network=network,
        bus_carrier=bus_carrier,
        output_dir=OUTPUT_DIR,
    )

    save_dataframe_csv(
        dispatch,
        OUTPUT_DIR / "dispatch_timeseries_mw.csv",
    )

    save_dataframe_csv(
        energy_balance_by_carrier,
        OUTPUT_DIR / "energy_balance_timeseries_mw.csv",
    )

    plot_annual_generation_mix(
        network=network,
        bus_carrier=bus_carrier,
        output_path=OUTPUT_DIR / "annual_electricity_generation_mix.png",
    )

    plot_capacity_bar(
        network=network,
        bus_carrier=bus_carrier,
        output_path=OUTPUT_DIR / "optimal_generation_capacity.png",
    )

    plot_capacity_factors_over_year(
        network=network,
        bus_carrier=bus_carrier,
        output_path=OUTPUT_DIR / "capacity_factors_over_year.png",
    )

    season_weeks = get_representative_weeks(network.snapshots)

    winter_week = season_weeks["winter"]
    summer_week = season_weeks["summer"]

    plot_energy_balance_week(
        network=network,
        balance_by_carrier_t=energy_balance_by_carrier,
        start_date=str(winter_week[0]),
        end_date=str(winter_week[-1]),
        output_path=OUTPUT_DIR / "energy_balance_winter_week.png",
    )

    plot_energy_balance_week(
        network=network,
        balance_by_carrier_t=energy_balance_by_carrier,
        start_date=str(summer_week[0]),
        end_date=str(summer_week[-1]),
        output_path=OUTPUT_DIR / "energy_balance_summer_week.png",
    )

    plot_duration_curves(
        network=network,
        bus_carrier=bus_carrier,
        output_path=OUTPUT_DIR / "dispatch_duration_curves.png",
    )

    print("\nTask 1 analysis finished.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()