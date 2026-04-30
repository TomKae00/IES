from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


# =========================================================
# PATH SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.helpers import (  # noqa: E402
    prepare_costs,
    load_country_timeseries,
)
from model.model import (  # noqa: E402
    create_network,
    custom_constraints,
)


# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = PROJECT_ROOT / "results" / "co2_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FONT_SIZE = 14

CARRIER_ALIASES = {
    "solar": ["solar"],
    "onshore wind": ["onwind", "onshore wind", "wind_onshore", "onshore"],
    "offshore wind": ["offwind", "offshore wind", "wind_offshore", "offshore"],
    "CCGT": ["gas CCGT", "CCGT", "ccgt", "gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
    "electricity": ["electricity", "AC"],
    "battery": ["battery"],
    "battery charger": ["battery charger", "battery_charger"],
    "battery discharger": [
        "battery discharger",
        "battery_discharge",
        "battery_discharger",
    ],
}

DEFAULT_CARRIER_COLORS = {
    "electricity": "#4C566A",
    "AC": "#4C566A",
    "solar": "#EBCB3B",
    "onshore wind": "#5AA469",
    "onwind": "#5AA469",
    "offshore wind": "#2E86AB",
    "offwind": "#2E86AB",
    "CCGT": "#D08770",
    "gas": "#D08770",
    "gas CCGT": "#D08770",
    "coal": "#5C5C5C",
    "nuclear": "#8F6BB3",
    "battery": "#E67E22",
    "battery charger": "#C06C84",
    "battery discharger": "#6C5B7B",
}

# Weekly energy-balance stacking order.
# Negative side: electricity, battery charger.
# Positive side bottom to top: nuclear, coal, CCGT, offshore, onshore, solar, battery discharger.
ENERGY_BALANCE_ORDER = [
    "electricity",
    "battery charger",
    "nuclear",
    "coal",
    "CCGT",
    "offshore wind",
    "onshore wind",
    "solar",
    "battery discharger",
]

POSITIVE_CARRIER_ORDER = [
    "nuclear",
    "coal",
    "CCGT",
    "offshore wind",
    "onshore wind",
    "solar",
    "battery discharger",
]


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
# GENERAL HELPERS
# =========================================================

def map_carrier_to_display_name(carrier: str) -> str:
    """
    Map internal carrier names to report-friendly labels.
    """
    for display_name, aliases in CARRIER_ALIASES.items():
        if carrier in aliases:
            return display_name

    return carrier


def get_carrier_color(carrier: str, n: pypsa.Network | None = None) -> str:
    """
    Return report color for a carrier.

    First checks n.carriers['color'] if available, then falls back to
    DEFAULT_CARRIER_COLORS.
    """
    display_name = map_carrier_to_display_name(carrier)

    candidates = [carrier, display_name]

    if display_name in CARRIER_ALIASES:
        candidates.extend(CARRIER_ALIASES[display_name])

    for candidate in candidates:
        if (
            n is not None
            and candidate in n.carriers.index
            and "color" in n.carriers.columns
            and pd.notna(n.carriers.at[candidate, "color"])
            and n.carriers.at[candidate, "color"] != ""
        ):
            return n.carriers.at[candidate, "color"]

        if candidate in DEFAULT_CARRIER_COLORS:
            return DEFAULT_CARRIER_COLORS[candidate]

    return "#999999"


def get_carrier_colors(
    carriers: pd.Index | list[str],
    n: pypsa.Network | None = None,
) -> list[str]:
    """
    Return colors for multiple carriers.
    """
    return [get_carrier_color(carrier, n=n) for carrier in carriers]


def drop_empty_carriers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all-zero and all-NaN columns.
    """
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, (df.fillna(0.0).abs() > 0.0).any(axis=0)]

    return df


def reorder_columns(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """
    Reorder columns according to a preferred order while keeping remaining columns.
    """
    ordered = [column for column in order if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]

    return df[ordered + remaining]


def carrier_order(columns: list[str] | pd.Index) -> list[str]:
    """
    Return preferred carrier order for capacity/generation plots.
    """
    return [c for c in POSITIVE_CARRIER_ORDER if c in columns] + [
        c for c in columns if c not in POSITIVE_CARRIER_ORDER
    ]


def get_electricity_bus_carrier(n: pypsa.Network) -> str:
    """
    Select the electricity bus carrier.
    """
    carriers = pd.Index(n.buses.carrier.dropna().unique())

    for candidate in ["AC", "electricity"]:
        if candidate in carriers:
            return candidate

    raise ValueError(
        f"Could not find an electricity bus carrier. Found bus carriers: {list(carriers)}"
    )


def add_target_line(
    ax: plt.Axes,
    target_cap_mt: float,
    label: str = "70% reduction target",
) -> None:
    """
    Add vertical 70% reduction target line.
    """
    ax.axvline(
        target_cap_mt,
        linestyle=":",
        linewidth=2.0,
        color="black",
        label=label,
    )


def format_co2_x_axis(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
) -> None:
    """
    Format x-axis with CO2 cap and reduction percentage.
    """
    ticks = plot_df["co2_cap_mt"].values
    labels = [
        f"{cap:.2f}\n{red:.0f}%"
        for cap, red in zip(
            plot_df["co2_cap_mt"],
            plot_df["co2_reduction_pct"],
        )
    ]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("CO$_2$ cap [Mt CO$_2$/year]\nReduction from baseline [%]")
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def prepare_plot_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare finite CO2 cap cases for plotting.
    """
    plot_df = results_df[np.isfinite(results_df["co2_cap_mt"])].copy()
    plot_df["co2_reduction_pct"] = (
        1.0 - plot_df["co2_cap_fraction"]
    ) * 100.0
    plot_df = plot_df.sort_values("co2_cap_mt")

    return plot_df


# =========================================================
# OPTIMIZATION
# =========================================================

def optimize_network(n: pypsa.Network, scenario: dict):
    """
    Optimize network with custom constraints.
    """
    def extra_functionality(network: pypsa.Network, snapshots):
        custom_constraints(network, snapshots, scenario)

    status = n.optimize(
        n.snapshots,
        extra_functionality=extra_functionality,
        solver_name="gurobi",
        solver_options={},
        include_objective_constant=False,
    )

    return status


# =========================================================
# RESULT EXTRACTION
# =========================================================

def calculate_network_emissions(n: pypsa.Network) -> float:
    """
    Return annual CO2 emissions in Mt CO2.

    Uses:
    generator dispatch [MWh_el] / efficiency * carrier CO2 intensity
    [tCO2/MWh_th].
    """
    if n.objective is None:
        raise ValueError("Network has not been optimized.")

    total_tco2 = 0.0

    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]

        if gen not in n.generators_t.p.columns:
            continue

        if carrier not in n.carriers.index:
            continue

        if "co2_emissions" in n.carriers.columns:
            co2_intensity = n.carriers.at[carrier, "co2_emissions"]
        else:
            co2_intensity = 0.0

        if pd.isna(co2_intensity):
            co2_intensity = 0.0

        efficiency = n.generators.at[gen, "efficiency"]

        if pd.isna(efficiency) or efficiency <= 0:
            efficiency = 1.0

        dispatch_mwh = n.generators_t.p[gen].clip(lower=0.0)
        primary_energy_mwh = dispatch_mwh / efficiency
        total_tco2 += (primary_energy_mwh * co2_intensity).sum()

    return total_tco2 / 1e6


def get_generation_by_carrier(n: pypsa.Network) -> dict:
    """
    Return annual generation by report-friendly carrier name in MWh.
    """
    generation = {}

    for gen in n.generators.index:
        carrier = map_carrier_to_display_name(n.generators.at[gen, "carrier"])

        if gen in n.generators_t.p.columns:
            generation[carrier] = (
                generation.get(carrier, 0.0)
                + n.generators_t.p[gen].clip(lower=0.0).sum()
            )

    return generation


def get_capacity_by_carrier(n: pypsa.Network) -> dict:
    """
    Return optimized generation capacity by report-friendly carrier name in MW.
    """
    capacities = {}

    for gen in n.generators.index:
        carrier = map_carrier_to_display_name(n.generators.at[gen, "carrier"])

        if "p_nom_opt" in n.generators.columns:
            cap = n.generators.at[gen, "p_nom_opt"]
        else:
            cap = n.generators.at[gen, "p_nom"]

        capacities[carrier] = capacities.get(carrier, 0.0) + cap

    return capacities


def get_battery_capacities(n: pypsa.Network) -> dict:
    """
    Flexible battery extraction.
    """
    result = {
        "battery_energy_mwh": 0.0,
        "battery_charger_mw": 0.0,
        "battery_discharger_mw": 0.0,
    }

    if len(n.stores) > 0:
        battery_stores = n.stores[
            n.stores.index.str.contains("battery", case=False, regex=False)
            | n.stores.carrier.astype(str).str.contains(
                "battery",
                case=False,
                regex=False,
            )
        ]

        if not battery_stores.empty:
            if "e_nom_opt" in battery_stores.columns:
                result["battery_energy_mwh"] = battery_stores["e_nom_opt"].sum()
            else:
                result["battery_energy_mwh"] = battery_stores["e_nom"].sum()

    if len(n.links) > 0:
        battery_links = n.links[
            n.links.index.str.contains("battery", case=False, regex=False)
            | n.links.carrier.astype(str).str.contains(
                "battery",
                case=False,
                regex=False,
            )
        ]

        chargers = battery_links[
            battery_links.index.str.contains("charger", case=False, regex=False)
        ]

        dischargers = battery_links[
            battery_links.index.str.contains("discharger", case=False, regex=False)
        ]

        if not chargers.empty:
            result["battery_charger_mw"] = chargers["p_nom_opt"].sum()

        if not dischargers.empty:
            result["battery_discharger_mw"] = dischargers["p_nom_opt"].sum()

    return result


def build_result_entry(
    n: pypsa.Network,
    co2_cap_fraction: float,
    co2_cap_mt: float,
    actual_emissions_mt: float,
) -> dict:
    """
    Build one result row.
    """
    result = {
        "co2_cap_fraction": co2_cap_fraction,
        "co2_cap_mt": co2_cap_mt,
        "system_cost_eur": n.objective,
        "actual_emissions_mt": actual_emissions_mt,
    }

    for carrier, cap in get_capacity_by_carrier(n).items():
        result[f"capacity_{carrier}_mw"] = cap

    for carrier, gen in get_generation_by_carrier(n).items():
        result[f"generation_{carrier}_mwh"] = gen

    result.update(get_battery_capacities(n))

    return result


# =========================================================
# ENERGY BALANCE
# =========================================================

def get_energy_balance_by_carrier(
    n: pypsa.Network,
    bus_carrier: str | None = None,
) -> pd.DataFrame:
    """
    Return time-resolved energy balance by carrier at the electricity bus carrier.

    Positive values are injections.
    Negative values are withdrawals.
    """
    if bus_carrier is None:
        bus_carrier = get_electricity_bus_carrier(n)

    balance = n.statistics.energy_balance(
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

    if len(balance_by_carrier_t.index) == len(n.snapshots):
        balance_by_carrier_t.index = n.snapshots

    balance_by_carrier_t = balance_by_carrier_t.rename(
        columns=lambda carrier: map_carrier_to_display_name(carrier)
    )

    balance_by_carrier_t = balance_by_carrier_t.T.groupby(level=0).sum().T
    balance_by_carrier_t = drop_empty_carriers(balance_by_carrier_t)
    balance_by_carrier_t = reorder_columns(
        balance_by_carrier_t,
        ENERGY_BALANCE_ORDER,
    )

    return balance_by_carrier_t


# =========================================================
# WEEKLY DISPATCH PLOTS
# =========================================================

def plot_weekly_dispatch(
    n: pypsa.Network,
    week_start: str = "2016-01-01",
    output_name: str = "weekly_dispatch",
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Plot weekly energy balance using n.statistics.energy_balance().
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    week_start_dt = pd.to_datetime(week_start)
    week_end_dt = week_start_dt + pd.Timedelta(days=7)

    balance = get_energy_balance_by_carrier(n)

    week_balance = balance.loc[
        (balance.index >= week_start_dt) & (balance.index < week_end_dt)
    ].copy()

    if week_balance.empty:
        print(f"No snapshots found for week starting {week_start_dt}")
        return

    week_balance = drop_empty_carriers(week_balance)
    week_balance = reorder_columns(week_balance, ENERGY_BALANCE_ORDER)

    colors = get_carrier_colors(week_balance.columns, n=n)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    week_balance.plot.area(
        ax=ax,
        stacked=True,
        linewidth=0.0,
        color=colors,
    )

    ax.set_ylabel("Power balance [MW]")
    ax.set_xlabel("Time")
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    y_max = max(
        abs(float(week_balance.min().min())),
        abs(float(week_balance.max().max())),
    )
    y_max = int(np.ceil(y_max / 500.0) * 500.0)
    ax.set_ylim(-y_max, y_max)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=REPORT_FONT_SIZE,
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    output_path = output_dir / f"{output_name}.png"
    save_figure(fig, output_path)
    plt.close(fig)

    print(f"Saved dispatch plot to {output_path}")


# =========================================================
# CO2 SENSITIVITY PLOTS
# =========================================================

def plot_co2_sensitivity_overview(
    results_df: pd.DataFrame,
    n_baseline: pypsa.Network,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Plot the original 2x2 overview figure, but with cleaner report styling.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = prepare_plot_df(results_df)
    baseline = results_df[results_df["co2_cap_fraction"] == 1.0].iloc[0]

    target_fraction = 0.30
    target_cap_mt = baseline["actual_emissions_mt"] * target_fraction

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 9.0),
        constrained_layout=True,
    )

    # -----------------------------------------------------
    # Generation mix
    # -----------------------------------------------------
    ax = axes[0, 0]

    generation_cols = [
        c for c in plot_df.columns
        if c.startswith("generation_") and c.endswith("_mwh")
    ]

    generation_carriers = [
        c.replace("generation_", "").replace("_mwh", "")
        for c in generation_cols
    ]

    generation_carriers = carrier_order(generation_carriers)

    bottom = np.zeros(len(plot_df))

    for carrier in generation_carriers:
        col = f"generation_{carrier}_mwh"

        if col not in plot_df.columns:
            continue

        values = plot_df[col].fillna(0.0).values / 1000.0

        if np.all(values <= 1e-6):
            continue

        color = get_carrier_color(carrier, n=n_baseline)

        ax.fill_between(
            plot_df["co2_cap_mt"],
            bottom,
            bottom + values,
            alpha=0.85,
            label=carrier,
            color=color,
            linewidth=0.0,
        )

        bottom += values

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("Generation [GWh/year]")
    format_co2_x_axis(ax, plot_df)
    ax.legend(
        loc="upper right",
        fontsize=REPORT_FONT_SIZE - 1,
        frameon=True,
        framealpha=0.95,
    )

    # -----------------------------------------------------
    # System cost
    # -----------------------------------------------------
    ax = axes[0, 1]

    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["system_cost_eur"] / 1e9,
        marker="o",
        linewidth=2.0,
        color="black",
    )

    ax.axhline(
        baseline["system_cost_eur"] / 1e9,
        linestyle="--",
        linewidth=1.8,
        color="gray",
        label="Baseline",
    )

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("System cost [bn EUR/year]")
    format_co2_x_axis(ax, plot_df)
    ax.legend(
        loc="upper right",
        fontsize=REPORT_FONT_SIZE,
        frameon=True,
        framealpha=0.95,
    )

    # -----------------------------------------------------
    # Actual emissions
    # -----------------------------------------------------
    ax = axes[1, 0]

    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["actual_emissions_mt"],
        marker="o",
        linewidth=2.0,
        color="black",
        label="Actual emissions",
    )

    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["co2_cap_mt"],
        linestyle="--",
        linewidth=1.8,
        color="gray",
        label="CO$_2$ cap",
    )

    ax.axhline(
        baseline["actual_emissions_mt"],
        linestyle=":",
        linewidth=1.8,
        color="gray",
        label="Baseline",
    )

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("Emissions [Mt CO$_2$/year]")
    format_co2_x_axis(ax, plot_df)
    ax.legend(
        loc="upper left",
        fontsize=REPORT_FONT_SIZE,
        frameon=True,
        framealpha=0.95,
    )

    # -----------------------------------------------------
    # Capacity mix
    # -----------------------------------------------------
    ax = axes[1, 1]

    capacity_cols = [
        c for c in plot_df.columns
        if c.startswith("capacity_") and c.endswith("_mw")
    ]

    capacity_carriers = [
        c.replace("capacity_", "").replace("_mw", "")
        for c in capacity_cols
    ]

    capacity_carriers = carrier_order(capacity_carriers)

    for carrier in capacity_carriers:
        col = f"capacity_{carrier}_mw"

        if col not in plot_df.columns:
            continue

        values = plot_df[col].fillna(0.0).values

        if np.all(values <= 1e-6):
            continue

        color = get_carrier_color(carrier, n=n_baseline)

        ax.plot(
            plot_df["co2_cap_mt"],
            values,
            marker="o",
            linewidth=2.0,
            label=carrier,
            color=color,
        )

    if "battery_energy_mwh" in plot_df.columns:
        values = plot_df["battery_energy_mwh"].fillna(0.0).values

        if np.any(values > 1e-6):
            battery_color = get_carrier_color("battery", n=n_baseline)

            ax.plot(
                plot_df["co2_cap_mt"],
                values,
                marker="s",
                linewidth=2.0,
                label="battery energy [MWh]",
                color=battery_color,
            )

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("Capacity [MW] / Battery energy [MWh]")
    format_co2_x_axis(ax, plot_df)
    ax.legend(
        loc="upper right",
        fontsize=REPORT_FONT_SIZE - 1,
        frameon=True,
        framealpha=0.95,
    )

    output_path = output_dir / "co2_sensitivity_overview.png"
    save_figure(fig, output_path)
    plt.close(fig)

    print(f"Saved overview plot to {output_path}")


def plot_co2_sensitivity_split(
    results_df: pd.DataFrame,
    n_baseline: pypsa.Network,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Create cleaner separate CO2 sensitivity plots.

    Saves:
    - co2_sensitivity_cost_emissions.png / .pdf
    - co2_sensitivity_generation_mix.png / .pdf
    - co2_sensitivity_capacity_mix.png / .pdf
    - co2_sensitivity_battery_energy.png / .pdf
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = prepare_plot_df(results_df)
    baseline = results_df[results_df["co2_cap_fraction"] == 1.0].iloc[0]

    target_fraction = 0.30
    target_cap_mt = baseline["actual_emissions_mt"] * target_fraction

    # =====================================================
    # 1) COST + EMISSIONS
    # =====================================================
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)

    ax = axes[0]
    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["system_cost_eur"] / 1e9,
        marker="o",
        linewidth=2.0,
        color="black",
        label="System cost",
    )

    ax.axhline(
        baseline["system_cost_eur"] / 1e9,
        linestyle="--",
        linewidth=1.8,
        color="gray",
        label="Baseline",
    )

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("System cost [bn EUR/year]")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    ax = axes[1]
    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["actual_emissions_mt"],
        marker="o",
        linewidth=2.0,
        color="black",
        label="Actual emissions",
    )

    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["co2_cap_mt"],
        linestyle="--",
        linewidth=1.8,
        color="gray",
        label="CO$_2$ cap",
    )

    ax.axhline(
        baseline["actual_emissions_mt"],
        linestyle=":",
        linewidth=1.8,
        color="gray",
        label="Baseline emissions",
    )

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("Emissions [Mt CO$_2$/year]")
    format_co2_x_axis(ax, plot_df)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.tight_layout()
    save_figure(fig, output_dir / "co2_sensitivity_cost_emissions.png")
    plt.close(fig)

    # =====================================================
    # 2) GENERATION MIX
    # =====================================================
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    generation_cols = [
        c for c in plot_df.columns
        if c.startswith("generation_") and c.endswith("_mwh")
    ]

    generation_carriers = [
        c.replace("generation_", "").replace("_mwh", "")
        for c in generation_cols
    ]
    generation_carriers = carrier_order(generation_carriers)

    bottom = np.zeros(len(plot_df))

    for carrier in generation_carriers:
        col = f"generation_{carrier}_mwh"

        if col not in plot_df.columns:
            continue

        values = plot_df[col].fillna(0.0).values / 1000.0

        if np.all(values <= 1e-6):
            continue

        color = get_carrier_color(carrier, n=n_baseline)

        ax.fill_between(
            plot_df["co2_cap_mt"],
            bottom,
            bottom + values,
            color=color,
            alpha=0.85,
            linewidth=0.0,
            label=carrier,
        )
        bottom += values

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("Generation [GWh/year]")
    format_co2_x_axis(ax, plot_df)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, output_dir / "co2_sensitivity_generation_mix.png")
    plt.close(fig)

    # =====================================================
    # 3) CAPACITY MIX, MW only
    # =====================================================
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    capacity_cols = [
        c for c in plot_df.columns
        if c.startswith("capacity_") and c.endswith("_mw")
    ]

    capacity_carriers = [
        c.replace("capacity_", "").replace("_mw", "")
        for c in capacity_cols
    ]
    capacity_carriers = carrier_order(capacity_carriers)

    for carrier in capacity_carriers:
        col = f"capacity_{carrier}_mw"

        if col not in plot_df.columns:
            continue

        values = plot_df[col].fillna(0.0).values

        if np.all(values <= 1e-6):
            continue

        color = get_carrier_color(carrier, n=n_baseline)

        ax.plot(
            plot_df["co2_cap_mt"],
            values,
            marker="o",
            linewidth=2.0,
            color=color,
            label=carrier,
        )

    add_target_line(ax, target_cap_mt)
    ax.set_ylabel("Installed capacity [MW]")
    format_co2_x_axis(ax, plot_df)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, output_dir / "co2_sensitivity_capacity_mix.png")
    plt.close(fig)

    # =====================================================
    # 4) BATTERY ENERGY
    # =====================================================
    if "battery_energy_mwh" in plot_df.columns and np.any(
        plot_df["battery_energy_mwh"].fillna(0.0).values > 1e-6
    ):
        fig, ax = plt.subplots(figsize=(9.5, 5.0))

        ax.plot(
            plot_df["co2_cap_mt"],
            plot_df["battery_energy_mwh"].fillna(0.0).values,
            marker="o",
            linewidth=2.0,
            color=get_carrier_color("battery", n=n_baseline),
            label="Battery energy",
        )

        add_target_line(ax, target_cap_mt)
        ax.set_ylabel("Battery energy capacity [MWh]")
        format_co2_x_axis(ax, plot_df)
        ax.legend(loc="upper left", frameon=True, framealpha=0.95)

        fig.tight_layout()
        save_figure(fig, output_dir / "co2_sensitivity_battery_energy.png")
        plt.close(fig)

    print("Saved split CO2 sensitivity plots.")


# =========================================================
# ANALYSIS
# =========================================================

def run_co2_sensitivity_analysis(
    cost_file: str | Path,
    timeseries_file: str | Path,
    financial_parameters: dict,
    scenario_parameters: dict,
    co2_cap_fractions: list[float] | None = None,
    co2_price: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pypsa.Network]:
    """
    Run CO2 cap sensitivity analysis.
    """
    if co2_cap_fractions is None:
        co2_cap_fractions = [
            0.8,
            0.6,
            0.4,
            0.3,
            0.2,
            0.1,
        ]

    print("\n" + "=" * 80)
    print("CO2 SENSITIVITY ANALYSIS")
    print("=" * 80)

    cost_data = prepare_costs(
        cost_file=cost_file,
        financial_parameters=financial_parameters,
        number_of_years=financial_parameters["nyears"],
    )

    all_timeseries_data = {}

    for country_code in scenario_parameters["countries"]:
        all_timeseries_data[country_code] = load_country_timeseries(
            timeseries_file=timeseries_file,
            country_code=country_code,
            year=scenario_parameters["weather_year"],
        )

    scenario = {
        "name": scenario_parameters.get("name", "co2_sensitivity"),
        "weather_year": scenario_parameters["weather_year"],
        "countries": scenario_parameters["countries"],
        "with_battery_storage": scenario_parameters.get("with_battery_storage", True),
        "with_interconnectors": scenario_parameters.get("with_interconnectors", False),
        "with_ch4_network": scenario_parameters.get("with_ch4_network", False),
        "with_h2_network": scenario_parameters.get("with_h2_network", False),
        "with_heat_sector": scenario_parameters.get("with_heat_sector", False),
        "with_heat_storage": scenario_parameters.get("with_heat_storage", False),
        "co2_price": co2_price,
        "co2_limit": None,
    }

    results = []

    print("\nRunning baseline with no CO2 cap...")

    n_baseline = create_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        scenario=scenario,
        heat_timeseries=None,
    )

    optimize_network(n_baseline, scenario)

    baseline_emissions = calculate_network_emissions(n_baseline)

    print(f"Baseline cost: {n_baseline.objective:,.0f} EUR")
    print(f"Baseline emissions: {baseline_emissions:.3f} Mt CO2")

    results.append(
        build_result_entry(
            n=n_baseline,
            co2_cap_fraction=1.0,
            co2_cap_mt=np.inf,
            actual_emissions_mt=baseline_emissions,
        )
    )

    plot_weekly_dispatch(
        n=n_baseline,
        week_start=f"{scenario_parameters['weather_year']}-01-01",
        output_name="baseline_weekly_dispatch",
        output_dir=OUTPUT_DIR,
    )

    for i, fraction in enumerate(sorted(co2_cap_fractions, reverse=True), start=1):
        co2_cap_mt = baseline_emissions * fraction

        print("\n" + "-" * 80)
        print(f"[{i}/{len(co2_cap_fractions)}] CO2 cap: {fraction:.0%} of baseline")
        print(f"Cap: {co2_cap_mt:.3f} Mt CO2")
        print("-" * 80)

        scenario_with_cap = scenario.copy()
        scenario_with_cap["co2_limit"] = co2_cap_mt * 1e6

        n_scenario = create_network(
            cost_data=cost_data,
            all_timeseries_data=all_timeseries_data,
            scenario=scenario_with_cap,
            heat_timeseries=None,
        )

        optimize_network(n_scenario, scenario_with_cap)

        if n_scenario.objective is None or pd.isna(n_scenario.objective):
            print(f"Optimization failed or infeasible for CO2 cap {co2_cap_mt:.3f} Mt")
            continue

        actual_emissions = calculate_network_emissions(n_scenario)

        cost_increase_pct = (
            (n_scenario.objective - n_baseline.objective)
            / n_baseline.objective
            * 100.0
        )

        print(f"System cost: {n_scenario.objective:,.0f} EUR")
        print(f"Cost increase: {cost_increase_pct:+.1f}%")
        print(f"Actual emissions: {actual_emissions:.3f} Mt CO2")

        results.append(
            build_result_entry(
                n=n_scenario,
                co2_cap_fraction=fraction,
                co2_cap_mt=co2_cap_mt,
                actual_emissions_mt=actual_emissions,
            )
        )

        if np.isclose(fraction, 0.30):
            plot_weekly_dispatch(
                n=n_scenario,
                week_start=f"{scenario_parameters['weather_year']}-01-01",
                output_name="weekly_dispatch_70pct_co2_reduction",
                output_dir=OUTPUT_DIR,
            )

    results_df = pd.DataFrame(results)

    return results_df, cost_data, n_baseline


def print_summary(
    results_df: pd.DataFrame,
    cost_data: pd.DataFrame,
    n_baseline: pypsa.Network,
    co2_price: float,
) -> None:
    """
    Print compact summary.
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_row = results_df[
        results_df["co2_cap_fraction"] == 1.0
    ].iloc[0]

    target_fraction = 0.30
    target_candidates = results_df[
        np.isclose(results_df["co2_cap_fraction"], target_fraction)
    ]

    if target_candidates.empty:
        print("\nNo 70% reduction case found in results.")
        return

    target_row = target_candidates.iloc[0]

    print("\nCO2 EMISSIONS:")
    print(f"Baseline: {baseline_row['actual_emissions_mt']:.3f} Mt CO2")
    print(f"70% reduction: {target_row['actual_emissions_mt']:.3f} Mt CO2")

    def print_capacity_mix(row, title):
        print(f"\n{title}")
        print("-" * 40)

        capacity_cols = [
            c for c in row.index if c.startswith("capacity_") and c.endswith("_mw")
        ]

        ordered_carriers = carrier_order(
            [
                c.replace("capacity_", "").replace("_mw", "")
                for c in capacity_cols
            ]
        )

        total_capacity = sum(row[c] for c in capacity_cols if not pd.isna(row[c]))

        for carrier in ordered_carriers:
            col = f"capacity_{carrier}_mw"

            if col not in row.index:
                continue

            value = row[col] if not pd.isna(row[col]) else 0.0
            share = value / total_capacity * 100.0 if total_capacity > 0 else 0.0

            print(f"{carrier}: {value:.1f} MW ({share:.1f}%)")

        print("\nBattery:")
        print(f"Energy: {row.get('battery_energy_mwh', 0.0):.1f} MWh")
        print(f"Charging: {row.get('battery_charger_mw', 0.0):.1f} MW")
        print(f"Discharging: {row.get('battery_discharger_mw', 0.0):.1f} MW")

    print_capacity_mix(baseline_row, "Baseline capacity by carrier:")
    print_capacity_mix(target_row, "Capacity at 70% CO2 reduction:")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Main function.
    """
    set_report_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    financial_parameters = {
        "fill_values": 0.0,
        "r": 0.07,
        "nyears": 1,
        "year": 2025,
        "co2_price": 80.0,
    }

    scenario_parameters = {
        "weather_year": "2016",
        "with_battery_storage": True,
        "with_interconnectors": False,
        "countries": ["DK"],
    }

    file_paths = {
        "cost_file": PROJECT_ROOT / "data" / f"costs_{financial_parameters['year']}.csv",
        "timeseries_file": PROJECT_ROOT
        / "data"
        / "time_series_60min_singleindex_filtered_2015-2020.csv",
    }

    results_df, cost_data, n_baseline = run_co2_sensitivity_analysis(
        cost_file=file_paths["cost_file"],
        timeseries_file=file_paths["timeseries_file"],
        financial_parameters=financial_parameters,
        scenario_parameters=scenario_parameters,
        co2_cap_fractions=[0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
        co2_price=financial_parameters["co2_price"],
    )

    results_path = OUTPUT_DIR / "co2_sensitivity_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved results to {results_path}")

    # Keep the original 4-panel overview figure.
    plot_co2_sensitivity_overview(
        results_df=results_df,
        n_baseline=n_baseline,
        output_dir=OUTPUT_DIR,
    )

    # Also create cleaner split figures for the report.
    plot_co2_sensitivity_split(
        results_df=results_df,
        n_baseline=n_baseline,
        output_dir=OUTPUT_DIR,
    )

    print_summary(
        results_df=results_df,
        cost_data=cost_data,
        n_baseline=n_baseline,
        co2_price=financial_parameters["co2_price"],
    )

    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()