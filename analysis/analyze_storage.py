from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETWORK_DIR = PROJECT_ROOT / "results" / "networks"
OUTPUT_DIR = PROJECT_ROOT / "results" / "storage_analysis"

# Adjust these if your filenames are different
BASE_NETWORK_NAME = "base_DK_2016.nc"
STORAGE_NETWORK_NAME = "storage_DK_2016.nc"

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
    "AC": "#4C566A",
    "electricity": "#4C566A",
    "solar": "#EBCB3B",
    "onwind": "#5AA469",
    "onshore wind": "#5AA469",
    "offwind": "#2E86AB",
    "offshore wind": "#2E86AB",
    "gas CCGT": "#D08770",
    "CCGT": "#D08770",
    "coal": "#5C5C5C",
    "nuclear": "#8F6BB3",
    "battery": "#E67E22",
    "battery charger": "#C06C84",
    "battery discharger": "#6C5B7B",
}

# Stacking order for the energy-balance plots.
# Positive side: solar at bottom, then offshore wind, then CCGT.
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

SUMMARY_ORDER = [
    "solar",
    "onshore wind",
    "offshore wind",
    "CCGT",
    "coal",
    "nuclear",
]


# =========================================================
# PLOT STYLE
# =========================================================

def set_report_plot_style() -> None:
    """
    Set report-ready matplotlib style.
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def get_network_path(network_name: str) -> Path:
    """
    Return path to a solved network file.
    """
    return NETWORK_DIR / network_name


def load_network(network_path: Path) -> pypsa.Network:
    """
    Load a PyPSA network from NetCDF.
    """
    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    network = pypsa.Network(network_path)
    network.sanitize()

    return network


def get_electricity_bus_carrier(network: pypsa.Network) -> str:
    """
    Return the electricity bus carrier.

    In the storage case, the network also contains a battery bus, so we cannot
    assume that there is only one bus carrier.
    """
    carriers = pd.Index(network.buses.carrier.dropna().unique())

    for candidate in ["AC", "electricity"]:
        if candidate in carriers:
            return candidate

    raise ValueError(
        f"Could not find an electricity bus carrier. Found bus carriers: {list(carriers)}"
    )


def map_carrier_to_display_name(carrier: str) -> str:
    """
    Map model carrier names to report-friendly names.
    """
    for display_name, aliases in CARRIER_ALIASES.items():
        if carrier in aliases:
            return display_name

    return carrier


def get_color_lookup_candidates(carrier: str) -> list[str]:
    """
    Return possible carrier names for color lookup.
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


def get_carrier_colors(
    network: pypsa.Network,
    carriers: pd.Index | list[str],
) -> list[str]:
    """
    Return colors for carriers.
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


def drop_empty_carriers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all-zero and all-NaN columns.
    """
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, (df.fillna(0.0).abs() > 0.0).any(axis=0)]

    return df


def reorder_carriers(df: pd.DataFrame, carrier_order: list[str]) -> pd.DataFrame:
    """
    Reorder DataFrame columns according to preferred carrier order.
    """
    ordered = [carrier for carrier in carrier_order if carrier in df.columns]
    remaining = [carrier for carrier in df.columns if carrier not in ordered]

    return df[ordered + remaining]


def get_representative_weeks(
    snapshots: pd.DatetimeIndex,
) -> dict[str, pd.DatetimeIndex]:
    """
    Select one full winter week and one full summer week.
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

        raise ValueError(f"Could not find full week for months {months}.")

    return {
        "winter": first_full_week([12, 1, 2]),
        "summer": first_full_week([6, 7, 8]),
    }


# =========================================================
# ENERGY BALANCE
# =========================================================

def get_energy_balance_by_carrier(
    network: pypsa.Network,
    bus_carrier: str,
) -> pd.DataFrame:
    """
    Return time-resolved energy balance at the selected electricity bus carrier,
    aggregated by carrier using PyPSA statistics.
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

    # Sum columns that map to the same report-friendly carrier name.
    balance_by_carrier_t = balance_by_carrier_t.T.groupby(level=0).sum().T

    balance_by_carrier_t = drop_empty_carriers(balance_by_carrier_t)
    balance_by_carrier_t = reorder_carriers(
        balance_by_carrier_t,
        ENERGY_BALANCE_CARRIER_ORDER,
    )

    return balance_by_carrier_t


def plot_energy_balance_week(
    network: pypsa.Network,
    balance_by_carrier_t: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_path: Path,
) -> None:
    """
    Plot one representative week of the system energy balance.
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

    ax.set_ylabel("Power balance [MW]")
    ax.set_xlabel("Time")
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=REPORT_FONT_SIZE,
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


# =========================================================
# TABLE VALUES
# =========================================================

def extract_generator_capacity(network: pypsa.Network) -> pd.Series:
    """
    Return optimized generator capacities by report-friendly carrier name.
    """
    if network.generators.empty:
        return pd.Series(0.0, index=SUMMARY_ORDER, name="capacity_mw")

    if "p_nom_opt" not in network.generators.columns:
        raise KeyError("'p_nom_opt' not found in network.generators.")

    carrier_display_names = network.generators["carrier"].map(
        map_carrier_to_display_name
    )

    capacities = network.generators["p_nom_opt"].groupby(carrier_display_names).sum()
    capacities = capacities.reindex(SUMMARY_ORDER, fill_value=0.0)
    capacities.name = "capacity_mw"

    return capacities


def extract_battery_energy_capacity(network: pypsa.Network) -> float:
    """
    Return optimized battery energy capacity in MWh.

    This checks stores with carrier 'battery' first.
    """
    if network.stores.empty:
        return 0.0

    stores = network.stores.copy()

    battery_mask = stores["carrier"].fillna("").isin(
        ["battery", "Battery", "battery storage"]
    )

    # Fallback: also catch stores connected to a battery bus.
    if "bus" in stores.columns:
        store_bus_carriers = stores["bus"].map(network.buses["carrier"])
        battery_mask = battery_mask | store_bus_carriers.fillna("").isin(["battery"])

    battery_stores = stores.loc[battery_mask]

    if battery_stores.empty:
        return 0.0

    if "e_nom_opt" in battery_stores.columns:
        return float(battery_stores["e_nom_opt"].fillna(0.0).sum())

    if "e_nom" in battery_stores.columns:
        return float(battery_stores["e_nom"].fillna(0.0).sum())

    return 0.0


def extract_battery_power_capacity(network: pypsa.Network) -> float:
    """
    Return optimized battery power capacity in MW.

    This checks battery charger/discharger links and returns the larger
    installed power capacity. For a symmetric battery this is the battery
    power capacity.
    """
    if network.links.empty:
        return 0.0

    links = network.links.copy()
    link_carriers = links["carrier"].fillna("")

    battery_power_mask = link_carriers.isin(
        [
            "battery charger",
            "battery discharger",
            "battery",
            "Battery",
        ]
    )

    # Fallback: catch links connected to a battery bus.
    for bus_col in ["bus0", "bus1"]:
        if bus_col in links.columns:
            linked_bus_carriers = links[bus_col].map(network.buses["carrier"])
            battery_power_mask = battery_power_mask | linked_bus_carriers.fillna("").isin(
                ["battery"]
            )

    battery_links = links.loc[battery_power_mask]

    if battery_links.empty:
        return 0.0

    if "p_nom_opt" in battery_links.columns:
        by_carrier = battery_links.groupby("carrier")["p_nom_opt"].sum()

        charger_capacity = by_carrier.get("battery charger", 0.0)
        discharger_capacity = by_carrier.get("battery discharger", 0.0)

        if charger_capacity > 0.0 or discharger_capacity > 0.0:
            return float(max(charger_capacity, discharger_capacity))

        return float(battery_links["p_nom_opt"].fillna(0.0).max())

    if "p_nom" in battery_links.columns:
        by_carrier = battery_links.groupby("carrier")["p_nom"].sum()

        charger_capacity = by_carrier.get("battery charger", 0.0)
        discharger_capacity = by_carrier.get("battery discharger", 0.0)

        if charger_capacity > 0.0 or discharger_capacity > 0.0:
            return float(max(charger_capacity, discharger_capacity))

        return float(battery_links["p_nom"].fillna(0.0).max())

    return 0.0


def extract_objective_bn_eur(network: pypsa.Network) -> float:
    """
    Return objective value in bn EUR.
    """
    objective = getattr(network, "objective", None)

    if objective is None:
        return float("nan")

    return float(objective) / 1e9


def build_table_row(network: pypsa.Network, row_name: str) -> pd.Series:
    """
    Build one row for the battery comparison table.
    """
    generator_capacity = extract_generator_capacity(network)

    row = pd.Series(
        {
            "Solar [MW]": generator_capacity["solar"],
            "Onshore [MW]": generator_capacity["onshore wind"],
            "Offshore [MW]": generator_capacity["offshore wind"],
            "CCGT [MW]": generator_capacity["CCGT"],
            "Coal [MW]": generator_capacity["coal"],
            "Nuclear [MW]": generator_capacity["nuclear"],
            "Battery energy [MWh]": extract_battery_energy_capacity(network),
            "Battery power [MW]": extract_battery_power_capacity(network),
            "Objective [bn EUR]": extract_objective_bn_eur(network),
        },
        name=row_name,
    )

    return row


def build_battery_comparison_table(
    base_network: pypsa.Network | None,
    storage_network: pypsa.Network,
) -> pd.DataFrame:
    """
    Build comparison table for base case and storage case.
    """
    rows = []

    if base_network is not None:
        rows.append(build_table_row(base_network, "Base case"))

    rows.append(build_table_row(storage_network, "Storage case"))

    comparison = pd.DataFrame(rows)

    return comparison


def print_comparison_table(comparison: pd.DataFrame) -> None:
    """
    Print rounded comparison table and LaTeX-friendly row values.
    """
    print("\nBattery comparison table values:")
    print(comparison.round(3))

    print("\nLaTeX-style rows:")
    for row_name, row in comparison.iterrows():
        battery_energy = row["Battery energy [MWh]"]
        battery_power = row["Battery power [MW]"]

        battery_energy_text = (
            "--" if pd.isna(battery_energy) or battery_energy == 0.0
            else f"{battery_energy:.1f}"
        )
        battery_power_text = (
            "--" if pd.isna(battery_power) or battery_power == 0.0
            else f"{battery_power:.1f}"
        )

        print(
            f"{row_name} & "
            f"{row['Solar [MW]']:.1f} & "
            f"{row['Onshore [MW]']:.1f} & "
            f"{row['Offshore [MW]']:.1f} & "
            f"{row['CCGT [MW]']:.1f} & "
            f"{row['Coal [MW]']:.1f} & "
            f"{row['Nuclear [MW]']:.1f} & "
            f"{battery_energy_text} & "
            f"{battery_power_text} & "
            f"{row['Objective [bn EUR]']:.3f} \\\\"
        )


# =========================================================
# BATTERY PERFORMANCE
# =========================================================

def find_battery_components(network: pypsa.Network) -> dict[str, list[str]]:
    """
    Find battery stores, charger links and discharger links.
    """
    battery_stores = []

    if not network.stores.empty:
        store_carrier_mask = network.stores["carrier"].fillna("").isin(["battery"])

        store_bus_carriers = network.stores["bus"].map(network.buses["carrier"])
        store_bus_mask = store_bus_carriers.fillna("").isin(["battery"])

        battery_stores = list(network.stores.index[store_carrier_mask | store_bus_mask])

    battery_chargers = []
    battery_dischargers = []

    if not network.links.empty:
        battery_chargers = list(
            network.links.index[
                network.links["carrier"].fillna("").isin(["battery charger"])
            ]
        )

        battery_dischargers = list(
            network.links.index[
                network.links["carrier"].fillna("").isin(["battery discharger"])
            ]
        )

    return {
        "stores": battery_stores,
        "chargers": battery_chargers,
        "dischargers": battery_dischargers,
    }


def calculate_battery_performance(network: pypsa.Network) -> pd.DataFrame:
    """
    Calculate battery performance indicators for each battery in the network.

    The calculation assumes the usual model structure:
    - {country}_battery_store
    - {country}_battery_charger
    - {country}_battery_discharger
    """
    components = find_battery_components(network)

    rows = []

    for store in components["stores"]:
        prefix = store.replace("_battery_store", "")

        charger = f"{prefix}_battery_charger"
        discharger = f"{prefix}_battery_discharger"

        if charger not in network.links.index or discharger not in network.links.index:
            continue

        energy_capacity_mwh = network.stores.at[store, "e_nom_opt"]
        charge_power_mw = network.links.at[charger, "p_nom_opt"]
        discharge_power_mw = network.links.at[discharger, "p_nom_opt"]

        charge_input = network.links_t.p0[charger].clip(lower=0.0)
        discharge_output = (-network.links_t.p1[discharger]).clip(lower=0.0)
        state_of_charge = network.stores_t.e[store]

        annual_charged_mwh = charge_input.sum()
        annual_discharged_mwh = discharge_output.sum()

        if annual_charged_mwh > 0.0:
            operational_roundtrip_efficiency = (
                annual_discharged_mwh / annual_charged_mwh
            )
        else:
            operational_roundtrip_efficiency = 0.0

        if energy_capacity_mwh > 0.0:
            equivalent_full_cycles = annual_discharged_mwh / energy_capacity_mwh
            mean_soc_pu = state_of_charge.mean() / energy_capacity_mwh
            max_soc_pu = state_of_charge.max() / energy_capacity_mwh
            min_soc_pu = state_of_charge.min() / energy_capacity_mwh
        else:
            equivalent_full_cycles = 0.0
            mean_soc_pu = 0.0
            max_soc_pu = 0.0
            min_soc_pu = 0.0

        charging_hours = int((charge_input > 1e-6).sum())
        discharging_hours = int((discharge_output > 1e-6).sum())
        idle_hours = int(
            ((charge_input <= 1e-6) & (discharge_output <= 1e-6)).sum()
        )

        rows.append(
            {
                "battery": prefix,
                "energy_capacity_mwh": energy_capacity_mwh,
                "charge_power_mw": charge_power_mw,
                "discharge_power_mw": discharge_power_mw,
                "annual_charged_mwh": annual_charged_mwh,
                "annual_discharged_mwh": annual_discharged_mwh,
                "annual_charged_gwh": annual_charged_mwh / 1000.0,
                "annual_discharged_gwh": annual_discharged_mwh / 1000.0,
                "operational_roundtrip_efficiency": operational_roundtrip_efficiency,
                "equivalent_full_cycles": equivalent_full_cycles,
                "max_soc_mwh": state_of_charge.max(),
                "mean_soc_mwh": state_of_charge.mean(),
                "min_soc_mwh": state_of_charge.min(),
                "max_soc_pu": max_soc_pu,
                "mean_soc_pu": mean_soc_pu,
                "min_soc_pu": min_soc_pu,
                "max_charge_mw": charge_input.max(),
                "max_discharge_mw": discharge_output.max(),
                "charging_hours": charging_hours,
                "discharging_hours": discharging_hours,
                "idle_hours": idle_hours,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("battery")


def print_battery_performance(performance: pd.DataFrame) -> None:
    """
    Print battery performance summary and LaTeX rows.
    """
    print("\nBattery performance indicators:")

    if performance.empty:
        print("No battery performance data found.")
        return

    print(performance.round(3))

    print("\nLaTeX-style battery performance rows:")
    for battery, row in performance.iterrows():
        print(
            f"{battery} & "
            f"{row['energy_capacity_mwh']:.1f} & "
            f"{row['charge_power_mw']:.1f} & "
            f"{row['discharge_power_mw']:.1f} & "
            f"{row['annual_charged_gwh']:.1f} & "
            f"{row['annual_discharged_gwh']:.1f} & "
            f"{row['operational_roundtrip_efficiency']:.3f} & "
            f"{row['equivalent_full_cycles']:.1f} & "
            f"{row['mean_soc_pu']:.3f} \\\\"
        )


def print_battery_performance_latex_table(performance: pd.DataFrame) -> None:
    """
    Print a complete LaTeX table for the appendix.
    """
    if performance.empty:
        return

    print("\nComplete LaTeX appendix table:")
    print(
        r"""\begin{table}[H]
\centering
\caption{Battery performance indicators in the storage case.}
\label{tab:battery_performance_appendix}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{lrrrrrrrr}
\toprule
\textbf{Battery}
& \textbf{Energy}
& \textbf{Charge}
& \textbf{Discharge}
& \textbf{Charged}
& \textbf{Discharged}
& \textbf{RT eff.}
& \textbf{Cycles}
& \textbf{Mean SOC} \\
\textbf{}
& \textbf{[MWh]}
& \textbf{[MW]}
& \textbf{[MW]}
& \textbf{[GWh]}
& \textbf{[GWh]}
& \textbf{[-]}
& \textbf{[-]}
& \textbf{[-]} \\
\midrule"""
    )

    for battery, row in performance.iterrows():
        print(
            f"{battery} "
            f"& {row['energy_capacity_mwh']:.1f} "
            f"& {row['charge_power_mw']:.1f} "
            f"& {row['discharge_power_mw']:.1f} "
            f"& {row['annual_charged_gwh']:.1f} "
            f"& {row['annual_discharged_gwh']:.1f} "
            f"& {row['operational_roundtrip_efficiency']:.3f} "
            f"& {row['equivalent_full_cycles']:.1f} "
            f"& {row['mean_soc_pu']:.3f} \\\\"
        )

    print(
        r"""\bottomrule
\end{tabular}
\end{table}"""
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    set_report_plot_style()
    ensure_output_dir(OUTPUT_DIR)

    storage_network_path = get_network_path(STORAGE_NETWORK_NAME)
    storage_network = load_network(storage_network_path)

    base_network_path = get_network_path(BASE_NETWORK_NAME)

    if base_network_path.exists():
        base_network = load_network(base_network_path)
    else:
        base_network = None
        print(f"\nBase network not found, skipping base comparison: {base_network_path}")

    bus_carrier = get_electricity_bus_carrier(storage_network)

    print(f"Selected electricity bus carrier: {bus_carrier}")
    print(f"Loaded storage network: {storage_network_path}")

    balance_by_carrier_t = get_energy_balance_by_carrier(
        network=storage_network,
        bus_carrier=bus_carrier,
    )

    balance_by_carrier_t.to_csv(
        OUTPUT_DIR / "storage_energy_balance_timeseries_mw.csv"
    )

    season_weeks = get_representative_weeks(storage_network.snapshots)

    winter_week = season_weeks["winter"]
    summer_week = season_weeks["summer"]

    plot_energy_balance_week(
        network=storage_network,
        balance_by_carrier_t=balance_by_carrier_t,
        start_date=str(winter_week[0]),
        end_date=str(winter_week[-1]),
        output_path=OUTPUT_DIR / "storage_energy_balance_winter_week.png",
    )

    plot_energy_balance_week(
        network=storage_network,
        balance_by_carrier_t=balance_by_carrier_t,
        start_date=str(summer_week[0]),
        end_date=str(summer_week[-1]),
        output_path=OUTPUT_DIR / "storage_energy_balance_summer_week.png",
    )

    comparison = build_battery_comparison_table(
        base_network=base_network,
        storage_network=storage_network,
    )

    comparison.to_csv(OUTPUT_DIR / "battery_comparison_table.csv")
    print_comparison_table(comparison)

    battery_performance = calculate_battery_performance(storage_network)

    battery_performance.to_csv(
        OUTPUT_DIR / "battery_performance_summary.csv"
    )

    print_battery_performance(battery_performance)
    print_battery_performance_latex_table(battery_performance)

    print("\nStorage analysis finished.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()