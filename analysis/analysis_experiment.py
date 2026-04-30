"""
PyPSA model construction and run script for the IES course project.

This file contains:
- network construction
- carriers
- buses, loads, generators
- battery storage
- electricity interconnectors
- gas network
- heat sector coupling
- CO2 constraint
- optimization helper
- report-ready analysis plots

Scenario definitions are stored in scenarios.py.
Data loading and cost preparation are stored in helpers.py.
"""

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
    silence_gurobi_logger,
    prepare_costs,
    load_all_countries_timeseries,
    load_heat_timeseries,
    calculate_conventional_marginal_cost,
)

from model.scenarios import (  # noqa: E402
    FINANCIAL_PARAMETERS,
    FILE_PATHS,
    SCENARIOS,
)


# =========================================================
# CONFIG
# =========================================================

ACTIVE_SCENARIO = "sector_coupling"

OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FONT_SIZE = 14

CARRIER_ALIASES = {
    "electricity": ["electricity", "AC"],
    "solar": ["solar"],
    "onshore wind": ["onwind", "onshore wind"],
    "offshore wind": ["offwind", "offshore wind"],
    "CCGT": ["gas CCGT", "CCGT", "gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
    "battery": ["battery"],
    "battery charger": ["battery charger", "battery charge"],
    "battery discharger": ["battery discharger", "battery discharge"],
    "electrolysis": ["electrolysis"],
    "H2 turbine": ["H2 turbine"],
    "H2 fuel cell": ["H2 fuel cell"],
    "heat pump": ["heat pump", "heat pump electricity"],
    "resistive heater": ["resistive heater", "resistive heater electricity"],
    "gas boiler": ["gas boiler"],
    "heat storage charger": ["heat storage charger"],
    "heat storage discharger": ["heat storage discharger", "heat storage discharge"],
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
    "CH4": "#A35D3D",
    "CH4 supply": "#8C564B",
    "CH4 pipeline": "#7B4B2A",
    "H2": "#4DA3FF",
    "H2 store": "#7FDBFF",
    "H2 pipeline": "#2F6DB3",
    "electrolysis": "#3A86FF",
    "H2 turbine": "#6FA8DC",
    "H2 fuel cell": "#5E81AC",
    "heat": "#B48EAD",
    "heat pump": "#A3BE8C",
    "resistive heater": "#BF616A",
    "gas boiler": "#A35D3D",
    "heat storage": "#D8DEE9",
    "heat storage charger": "#88C0D0",
    "heat storage discharger": "#81A1C1",
}

# Electricity balance order.
# Negative side first: electricity demand, battery charging, electrolysis, heat-sector electricity use.
# Positive side bottom to top: nuclear, coal, CCGT, offshore wind, onshore wind, solar, storage/H2.
ELECTRICITY_BALANCE_ORDER = [
    "electricity",
    "battery charger",
    "electrolysis",
    "heat pump",
    "resistive heater",
    "nuclear",
    "coal",
    "CCGT",
    "offshore wind",
    "onshore wind",
    "solar",
    "battery discharger",
    "H2 turbine",
    "H2 fuel cell",
]

POSITIVE_ELECTRICITY_ORDER = [
    "nuclear",
    "coal",
    "CCGT",
    "offshore wind",
    "onshore wind",
    "solar",
    "battery discharger",
    "H2 turbine",
    "H2 fuel cell",
]

HEAT_BALANCE_ORDER = [
    "heat pump",
    "resistive heater",
    "gas boiler",
    "heat storage discharger",
]


# =========================================================
# PATH HELPERS
# =========================================================

def resolve_project_path(path_like: str | Path) -> Path:
    """
    Resolve a path relative to PROJECT_ROOT unless it is already absolute.
    """
    path = Path(path_like)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


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


def get_carrier_color(n: pypsa.Network, carrier: str) -> str:
    """
    Return color for carrier from n.carriers if available, otherwise from defaults.
    """
    display_name = map_carrier_to_display_name(carrier)
    candidates = [carrier, display_name]

    if display_name in CARRIER_ALIASES:
        candidates.extend(CARRIER_ALIASES[display_name])

    for candidate in candidates:
        if (
            candidate in n.carriers.index
            and "color" in n.carriers.columns
            and pd.notna(n.carriers.at[candidate, "color"])
            and n.carriers.at[candidate, "color"] != ""
        ):
            return n.carriers.at[candidate, "color"]

        if candidate in DEFAULT_CARRIER_COLORS:
            return DEFAULT_CARRIER_COLORS[candidate]

    return "#999999"


def get_carrier_colors(
    n: pypsa.Network,
    columns: pd.Index | list[str],
) -> list[str]:
    """
    Return colors for multiple carriers.
    """
    return [get_carrier_color(n, carrier) for carrier in columns]


def drop_empty_carriers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all-zero and all-NaN columns.
    """
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, (df.fillna(0.0).abs() > 0.0).any(axis=0)]

    return df


def reorder_columns(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """
    Reorder columns according to preferred order while keeping remaining columns.
    """
    ordered = [column for column in order if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]

    return df[ordered + remaining]


def get_electricity_bus_carrier(n: pypsa.Network) -> str:
    """
    Return electricity bus carrier.
    """
    carriers = pd.Index(n.buses.carrier.dropna().unique())

    for candidate in ["AC", "electricity"]:
        if candidate in carriers:
            return candidate

    raise ValueError(
        f"Could not find electricity bus carrier. Found bus carriers: {list(carriers)}"
    )


def get_peak_winter_week_index(
    n: pypsa.Network,
    load_name: str = "DK_electricity_demand",
) -> pd.DatetimeIndex:
    """
    Return the winter week with the highest average Danish electricity demand.
    """
    snapshots = n.snapshots
    winter_mask = (snapshots.month == 12) | (snapshots.month == 1)

    if load_name not in n.loads_t.p_set.columns:
        raise KeyError(
            f"Could not find load '{load_name}' in n.loads_t.p_set. "
            f"Available loads are: {list(n.loads_t.p_set.columns)}"
        )

    load = n.loads_t.p_set[load_name]
    winter_load = load.loc[winter_mask]

    weekly_avg_load = winter_load.resample("W").mean()
    max_load_week_end = weekly_avg_load.idxmax()

    week_start = max_load_week_end - pd.Timedelta(days=6)
    week_end = max_load_week_end

    return load.loc[week_start:week_end].index


# =========================================================
# CARRIERS
# =========================================================

def add_carriers(n: pypsa.Network) -> None:
    """
    Add all carriers used across possible scenarios.
    """
    carrier_data = {
        "AC": {"color": "#4C566A"},
        "solar": {"color": "#EBCB3B"},
        "onwind": {"color": "#5AA469"},
        "offwind": {"color": "#2E86AB"},
        "gas CCGT": {"color": "#D08770"},
        "coal": {"color": "#5C5C5C"},
        "nuclear": {"color": "#8F6BB3"},
        "battery": {"color": "#E67E22"},
        "battery charger": {"color": "#C06C84"},
        "battery discharger": {"color": "#6C5B7B"},
        "CH4": {"color": "#A35D3D"},
        "CH4 supply": {"color": "#8C564B"},
        "CH4 pipeline": {"color": "#7B4B2A"},
        "CCGT": {"color": "#D08770"},
        "H2": {"color": "#4DA3FF"},
        "H2 store": {"color": "#7FDBFF"},
        "H2 pipeline": {"color": "#2F6DB3"},
        "electrolysis": {"color": "#3A86FF"},
        "H2 turbine": {"color": "#6FA8DC"},
        "H2 fuel cell": {"color": "#5E81AC"},
        "heat": {"color": "#B48EAD"},
        "heat pump": {"color": "#A3BE8C"},
        "resistive heater": {"color": "#BF616A"},
        "gas boiler": {"color": "#A35D3D"},
        "heat storage": {"color": "#D8DEE9"},
        "heat storage charger": {"color": "#88C0D0"},
        "heat storage discharger": {"color": "#81A1C1"},
    }

    for carrier, attrs in carrier_data.items():
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier, **attrs)


def set_carrier_co2_emissions(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
) -> None:
    """
    Set carrier CO2 emissions for optional global CO2 constraints.
    """
    if "co2_emissions" not in n.carriers.columns:
        n.carriers["co2_emissions"] = 0.0

    if "gas" in cost_data.index and "CO2 intensity" in cost_data.columns:
        gas_co2_intensity = cost_data.at["gas", "CO2 intensity"]
        n.carriers.loc["gas CCGT", "co2_emissions"] = gas_co2_intensity
        n.carriers.loc["CH4 supply", "co2_emissions"] = gas_co2_intensity
        n.carriers.loc["gas boiler", "co2_emissions"] = gas_co2_intensity

    if "coal" in cost_data.index and "CO2 intensity" in cost_data.columns:
        n.carriers.loc["coal", "co2_emissions"] = cost_data.at[
            "coal",
            "CO2 intensity",
        ]


# =========================================================
# BASE ELECTRICITY MODEL
# =========================================================

def add_electricity(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
    all_timeseries_data: dict[str, dict[str, pd.Series]],
    scenario: dict,
) -> None:
    """
    Add electricity buses, loads, renewable generators, and conventional generators.
    """
    for country_code in scenario["countries"]:
        timeseries_data = all_timeseries_data[country_code]

        n.add("Bus", country_code, carrier="AC")

        n.add(
            "Load",
            f"{country_code}_electricity_demand",
            bus=country_code,
            carrier="AC",
            p_set=timeseries_data["load"],
        )

        n.add(
            "Generator",
            f"{country_code}_solar",
            bus=country_code,
            carrier="solar",
            p_nom_extendable=True,
            p_max_pu=timeseries_data["solar_cf"],
            capital_cost=cost_data.at["solar", "fixed"],
            marginal_cost=cost_data.at["solar", "VOM"],
        )

        n.add(
            "Generator",
            f"{country_code}_onshore_wind",
            bus=country_code,
            carrier="onwind",
            p_nom_extendable=True,
            p_max_pu=timeseries_data["onshore_wind_cf"],
            capital_cost=cost_data.at["onwind", "fixed"],
            marginal_cost=cost_data.at["onwind", "VOM"],
        )

        n.add(
            "Generator",
            f"{country_code}_offshore_wind",
            bus=country_code,
            carrier="offwind",
            p_nom_extendable=True,
            p_max_pu=timeseries_data["offshore_wind_cf"],
            capital_cost=cost_data.at["offwind", "fixed"],
            marginal_cost=cost_data.at["offwind", "VOM"],
        )

        if (
            not scenario.get("with_ch4_network", False)
            and not scenario.get("with_h2_network", False)
        ):
            marginal_cost = calculate_conventional_marginal_cost(
                cost_data=cost_data,
                technology="CCGT",
                co2_price=scenario["co2_price"],
            )

            n.add(
                "Generator",
                f"{country_code}_CCGT",
                bus=country_code,
                carrier="gas CCGT",
                p_nom_extendable=True,
                capital_cost=cost_data.at["CCGT", "fixed"],
                marginal_cost=marginal_cost,
                efficiency=cost_data.at["CCGT", "efficiency"],
            )

        marginal_cost = calculate_conventional_marginal_cost(
            cost_data=cost_data,
            technology="coal",
            co2_price=scenario["co2_price"],
        )

        n.add(
            "Generator",
            f"{country_code}_coal",
            bus=country_code,
            carrier="coal",
            p_nom_extendable=True,
            capital_cost=cost_data.at["coal", "fixed"],
            marginal_cost=marginal_cost,
            efficiency=cost_data.at["coal", "efficiency"],
        )

        if "nuclear" in cost_data.index:
            marginal_cost = calculate_conventional_marginal_cost(
                cost_data=cost_data,
                technology="nuclear",
                co2_price=scenario["co2_price"],
            )

            n.add(
                "Generator",
                f"{country_code}_nuclear",
                bus=country_code,
                carrier="nuclear",
                p_nom_extendable=True,
                capital_cost=cost_data.at["nuclear", "fixed"],
                marginal_cost=marginal_cost,
                efficiency=cost_data.at["nuclear", "efficiency"],
                p_min_pu=0.5,
                ramp_limit_up=0.3,
                ramp_limit_down=0.3,
            )


# =========================================================
# STORAGE EXTENSION
# =========================================================

def add_battery_storage(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
    scenario: dict,
) -> None:
    """
    Add battery storage to each modeled country.
    """
    for country_code in scenario["countries"]:
        battery_bus = f"{country_code}_battery"

        n.add("Bus", battery_bus, carrier="battery")

        n.add(
            "Store",
            f"{country_code}_battery_store",
            bus=battery_bus,
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=cost_data.at["battery storage", "fixed"],
        )

        n.add(
            "Link",
            f"{country_code}_battery_charger",
            bus0=country_code,
            bus1=battery_bus,
            carrier="battery charger",
            efficiency=cost_data.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=cost_data.at["battery inverter", "fixed"],
            marginal_cost=cost_data.at["battery inverter", "VOM"],
            p_nom_extendable=True,
        )

        n.add(
            "Link",
            f"{country_code}_battery_discharger",
            bus0=battery_bus,
            bus1=country_code,
            carrier="battery discharger",
            efficiency=cost_data.at["battery inverter", "efficiency"] ** 0.5,
            marginal_cost=cost_data.at["battery inverter", "VOM"],
            p_nom_extendable=True,
        )


# =========================================================
# ELECTRICITY INTERCONNECTOR EXTENSION
# =========================================================

def add_interconnectors(n: pypsa.Network) -> None:
    """
    Add fixed bidirectional electricity interconnectors.
    """
    line_data = [
        ("DK", "NO", "DK_NO", 1632.0),
        ("DK", "SE", "DK_SE", 2415.0),
        ("DK", "DE", "DK_DE", 3500.0),
        ("DE", "SE", "DE_SE", 615.0),
        ("NO", "SE", "NO_SE", 3645.0),
        ("DE", "NO", "DE_NO", 1400.0),
    ]

    for bus0, bus1, name, capacity in line_data:
        if bus0 not in n.buses.index or bus1 not in n.buses.index:
            continue

        n.add(
            "Line",
            name,
            bus0=bus0,
            bus1=bus1,
            s_nom=capacity,
            s_nom_extendable=False,
            x=0.1,
            r=0.1,
        )


# =========================================================
# GAS NETWORK EXTENSION
# =========================================================

def add_gas(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
    scenario: dict,
) -> None:
    """
    Add CH4 and/or H2 sector components.
    """
    countries = scenario["countries"]

    with_ch4_network = scenario.get("with_ch4_network", True)
    with_h2_network = scenario.get("with_h2_network", True)

    if not with_ch4_network and not with_h2_network:
        raise ValueError(
            "add_gas() was called although both CH4 and H2 networks are disabled."
        )

    for country_code in countries:
        if with_ch4_network:
            n.add("Bus", f"{country_code}_CH4", carrier="CH4")

            n.add(
                "Link",
                f"{country_code}_CCGT",
                bus0=f"{country_code}_CH4",
                bus1=country_code,
                carrier="CCGT",
                p_nom_extendable=True,
                efficiency=cost_data.at["CCGT", "efficiency"],
                capital_cost=cost_data.at["CCGT", "fixed"],
                marginal_cost=cost_data.at["CCGT", "VOM"],
            )

        if with_h2_network:
            n.add("Bus", f"{country_code}_H2", carrier="H2")

            n.add(
                "Link",
                f"{country_code}_electrolyzer",
                bus0=country_code,
                bus1=f"{country_code}_H2",
                carrier="electrolysis",
                p_nom_extendable=True,
                efficiency=cost_data.at["electrolysis", "efficiency"],
                capital_cost=cost_data.at["electrolysis", "fixed"],
                marginal_cost=cost_data.at["electrolysis", "VOM"],
            )

            n.add(
                "Link",
                f"{country_code}_H2_turbine",
                bus0=f"{country_code}_H2",
                bus1=country_code,
                carrier="H2 turbine",
                p_nom_extendable=True,
                efficiency=cost_data.at["OCGT", "efficiency"],
                capital_cost=cost_data.at["OCGT", "fixed"],
                marginal_cost=cost_data.at["OCGT", "VOM"],
            )

            n.add(
                "Link",
                f"{country_code}_H2_fuel_cell",
                bus0=f"{country_code}_H2",
                bus1=country_code,
                carrier="H2 fuel cell",
                p_nom_extendable=True,
                efficiency=cost_data.at["fuel cell", "efficiency"],
                capital_cost=cost_data.at["fuel cell", "fixed"],
                marginal_cost=cost_data.at["fuel cell", "VOM"],
            )

            n.add(
                "Store",
                f"{country_code}_H2_store",
                bus=f"{country_code}_H2",
                carrier="H2 store",
                e_nom_extendable=True,
                e_cyclic=True,
                capital_cost=cost_data.at[
                    "hydrogen storage tank type 1 including compressor",
                    "fixed",
                ],
            )

    if with_ch4_network:
        if "NO" in countries:
            fuel_cost = cost_data.at["gas", "fuel"]

            if "CO2 intensity" in cost_data.columns:
                co2_intensity = cost_data.at["gas", "CO2 intensity"]
                co2_cost = co2_intensity * scenario["co2_price"]
            else:
                co2_cost = 0.0

            n.add(
                "Generator",
                "NO_CH4_supply",
                bus="NO_CH4",
                carrier="CH4 supply",
                p_nom_extendable=True,
                capital_cost=0.0,
                marginal_cost=fuel_cost + co2_cost,
            )
        else:
            print("Warning: NO not in modeled countries. No CH4 supply added.")

    gas_corridor_data = [
        ("NO", "DK", "NO_DK", 600.0),
        ("NO", "SE", "NO_SE", 500.0),
        ("NO", "DE", "NO_DE", 900.0),
        ("DK", "SE", "DK_SE", 300.0),
        ("DK", "DE", "DK_DE", 250.0),
        ("DE", "SE", "DE_SE", 700.0),
    ]

    for country_a, country_b, corridor_name, length_km in gas_corridor_data:
        if country_a not in countries or country_b not in countries:
            print(
                f"Skipping gas corridor {corridor_name}: "
                f"{country_a} or {country_b} not in network."
            )
            continue

        if with_ch4_network:
            ch4_tech_name = "CH4 (g) pipeline"
            ch4_electricity_input = cost_data.at[ch4_tech_name, "electricity-input"]

            ch4_efficiency = 1.0 - ch4_electricity_input * length_km / 1000.0
            ch4_efficiency = max(ch4_efficiency, 0.0)

            n.add(
                "Link",
                f"{corridor_name}_CH4_{country_a}_to_{country_b}",
                bus0=f"{country_a}_CH4",
                bus1=f"{country_b}_CH4",
                carrier="CH4 pipeline",
                p_nom_extendable=True,
                efficiency=ch4_efficiency,
                marginal_cost=0.0,
            )

            n.add(
                "Link",
                f"{corridor_name}_CH4_{country_b}_to_{country_a}",
                bus0=f"{country_b}_CH4",
                bus1=f"{country_a}_CH4",
                carrier="CH4 pipeline",
                p_nom_extendable=True,
                efficiency=ch4_efficiency,
                marginal_cost=0.0,
            )

        if with_h2_network:
            h2_tech_name = "H2 (g) pipeline"
            h2_electricity_input = cost_data.at[h2_tech_name, "electricity-input"]

            h2_efficiency = 1.0 - h2_electricity_input * length_km / 1000.0
            h2_efficiency = max(h2_efficiency, 0.0)

            n.add(
                "Link",
                f"{corridor_name}_H2_{country_a}_to_{country_b}",
                bus0=f"{country_a}_H2",
                bus1=f"{country_b}_H2",
                carrier="H2 pipeline",
                p_nom_extendable=True,
                efficiency=h2_efficiency,
                marginal_cost=0.0,
            )

            n.add(
                "Link",
                f"{corridor_name}_H2_{country_b}_to_{country_a}",
                bus0=f"{country_b}_H2",
                bus1=f"{country_a}_H2",
                carrier="H2 pipeline",
                p_nom_extendable=True,
                efficiency=h2_efficiency,
                marginal_cost=0.0,
            )


# =========================================================
# SECTOR COUPLING EXTENSION: HEAT
# =========================================================

def add_heat(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
    scenario: dict,
    heat_timeseries: dict[str, dict[str, pd.Series]],
) -> None:
    """
    Add a decentral heat sector to all modeled countries.
    """
    with_ch4_network = scenario.get("with_ch4_network", False)

    for country_code in scenario["countries"]:
        if country_code not in heat_timeseries:
            raise KeyError(f"No heat time series found for {country_code}.")

        heat_demand = heat_timeseries[country_code]["heat_demand"]
        heat_pump_cop = heat_timeseries[country_code]["ashp_cop"]

        heat_bus = f"{country_code}_heat"
        heat_storage_bus = f"{country_code}_heat_storage"

        n.add("Bus", heat_bus, carrier="heat")

        n.add(
            "Load",
            f"{country_code}_heat_demand",
            bus=heat_bus,
            carrier="heat",
            p_set=heat_demand,
        )

        heat_pump_tech = "decentral air-sourced heat pump"

        n.add(
            "Link",
            f"{country_code}_ASHP",
            bus0=country_code,
            bus1=heat_bus,
            carrier="heat pump",
            p_nom_extendable=True,
            efficiency=heat_pump_cop,
            capital_cost=cost_data.at[heat_pump_tech, "fixed"],
            marginal_cost=cost_data.at[heat_pump_tech, "VOM"],
        )

        resistive_heater_tech = "decentral resistive heater"

        n.add(
            "Link",
            f"{country_code}_resistive_heater",
            bus0=country_code,
            bus1=heat_bus,
            carrier="resistive heater",
            p_nom_extendable=True,
            efficiency=cost_data.at[resistive_heater_tech, "efficiency"],
            capital_cost=cost_data.at[resistive_heater_tech, "fixed"],
            marginal_cost=cost_data.at[resistive_heater_tech, "VOM"],
        )

        gas_boiler_tech = "decentral gas boiler"
        gas_boiler_efficiency = cost_data.at[gas_boiler_tech, "efficiency"]

        if with_ch4_network:
            n.add(
                "Link",
                f"{country_code}_gas_boiler",
                bus0=f"{country_code}_CH4",
                bus1=heat_bus,
                carrier="gas boiler",
                p_nom_extendable=True,
                efficiency=gas_boiler_efficiency,
                capital_cost=cost_data.at[gas_boiler_tech, "fixed"],
                marginal_cost=cost_data.at[gas_boiler_tech, "VOM"],
            )

        else:
            gas_fuel_cost = cost_data.at["gas", "fuel"]

            if "CO2 intensity" in cost_data.columns:
                gas_co2_intensity = cost_data.at["gas", "CO2 intensity"]
                gas_boiler_co2_cost = (
                    gas_co2_intensity / gas_boiler_efficiency * scenario["co2_price"]
                )
            else:
                gas_boiler_co2_cost = 0.0

            gas_boiler_marginal_cost = (
                gas_fuel_cost / gas_boiler_efficiency
                + cost_data.at[gas_boiler_tech, "VOM"]
                + gas_boiler_co2_cost
            )

            n.add(
                "Generator",
                f"{country_code}_gas_boiler",
                bus=heat_bus,
                carrier="gas boiler",
                p_nom_extendable=True,
                efficiency=gas_boiler_efficiency,
                capital_cost=cost_data.at[gas_boiler_tech, "fixed"],
                marginal_cost=gas_boiler_marginal_cost,
            )

        if scenario.get("with_heat_storage", False):
            water_tank_storage_tech = "decentral water tank storage"
            water_tank_charger_tech = "decentral water tank charger"
            water_tank_discharger_tech = "decentral water tank discharger"

            n.add("Bus", heat_storage_bus, carrier="heat storage")

            n.add(
                "Link",
                f"{country_code}_water_tank_charger",
                bus0=heat_bus,
                bus1=heat_storage_bus,
                carrier="heat storage charger",
                p_nom_extendable=True,
                efficiency=cost_data.at[water_tank_charger_tech, "efficiency"],
                marginal_cost=cost_data.at[water_tank_charger_tech, "VOM"],
            )

            n.add(
                "Link",
                f"{country_code}_water_tank_discharger",
                bus0=heat_storage_bus,
                bus1=heat_bus,
                carrier="heat storage discharger",
                p_nom_extendable=True,
                efficiency=cost_data.at[water_tank_discharger_tech, "efficiency"],
                marginal_cost=cost_data.at[water_tank_discharger_tech, "VOM"],
            )

            n.links.loc[
                f"{country_code}_water_tank_charger",
                "energy_to_power_ratio",
            ] = cost_data.at[
                water_tank_storage_tech,
                "energy to power ratio",
            ]

            n.add(
                "Store",
                f"{country_code}_water_tank_store",
                bus=heat_storage_bus,
                carrier="heat storage",
                e_nom_extendable=True,
                e_cyclic=True,
                standing_loss=cost_data.at[
                    water_tank_storage_tech,
                    "standing losses",
                ] / 100,
                capital_cost=cost_data.at[water_tank_storage_tech, "fixed"],
            )


# =========================================================
# CO2 CONSTRAINTS
# =========================================================

def add_global_co2_constraint(
    n: pypsa.Network,
    co2_limit: float,
) -> None:
    """
    Add global CO2 constraint.
    """
    n.add(
        "GlobalConstraint",
        "CO2_limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_limit,
    )


# =========================================================
# CUSTOM OPTIMIZATION CONSTRAINTS
# =========================================================

def add_battery_charger_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add battery charger/discharger capacity ratio constraints.
    """
    if n.links.empty or not n.links.p_nom_extendable.any():
        return

    charger_links = n.links.index[
        n.links.index.str.contains("_battery_charger")
        & n.links.p_nom_extendable
    ]

    discharger_links = n.links.index[
        n.links.index.str.contains("_battery_discharger")
        & n.links.p_nom_extendable
    ]

    if charger_links.empty or discharger_links.empty:
        print(
            "No extendable battery charger/discharger links found. "
            "Skipping battery charger ratio constraints."
        )
        return

    for charger_link in charger_links:
        discharger_link = charger_link.replace(
            "_battery_charger",
            "_battery_discharger",
        )

        if discharger_link not in discharger_links:
            raise RuntimeError(
                f"Could not find matching battery discharger '{discharger_link}' "
                f"for charger link '{charger_link}'."
            )

        charger_p_nom = n.model["Link-p_nom"].loc[charger_link]
        discharger_p_nom = n.model["Link-p_nom"].loc[discharger_link]
        discharger_efficiency = n.links.at[discharger_link, "efficiency"]

        n.model.add_constraints(
            charger_p_nom - discharger_efficiency * discharger_p_nom == 0,
            name=f"battery_charger_ratio_{charger_link}",
        )


def add_tes_energy_to_power_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add thermal energy storage energy-to-power ratio constraints.
    """
    charger_links = n.links.index[
        n.links.index.str.contains("_water_tank_charger")
        & n.links.p_nom_extendable
    ]

    if charger_links.empty:
        print(
            "No extendable water tank charger links found. "
            "Skipping TES energy-to-power ratio constraints."
        )
        return

    if "energy_to_power_ratio" not in n.links.columns:
        raise KeyError(
            "Column 'energy_to_power_ratio' not found in n.links. "
            "Make sure it is assigned when adding the water tank charger."
        )

    for charger_link in charger_links:
        store_name = charger_link.replace(
            "_water_tank_charger",
            "_water_tank_store",
        )

        if store_name not in n.stores.index:
            raise RuntimeError(
                f"Could not find matching TES store '{store_name}' "
                f"for charger link '{charger_link}'."
            )

        if not n.stores.at[store_name, "e_nom_extendable"]:
            print(
                f"Store '{store_name}' is not extendable. "
                "Skipping TES energy-to-power ratio constraint."
            )
            continue

        energy_to_power_ratio = n.links.at[
            charger_link,
            "energy_to_power_ratio",
        ]

        charger_p_nom = n.model["Link-p_nom"].loc[charger_link]
        store_e_nom = n.model["Store-e_nom"].loc[store_name]

        n.model.add_constraints(
            store_e_nom - energy_to_power_ratio * charger_p_nom == 0,
            name=f"TES_energy_to_power_ratio_{store_name}",
        )


def custom_constraints(
    n: pypsa.Network,
    snapshots,
    scenario: dict,
) -> None:
    """
    Add custom optimization constraints depending on active scenario options.
    """
    if scenario.get("with_battery_storage", False):
        add_battery_charger_ratio_constraints(n)

    if scenario.get("with_heat_storage", False):
        add_tes_energy_to_power_ratio_constraints(n)


# =========================================================
# NETWORK CREATION
# =========================================================

def create_network(
    cost_data: pd.DataFrame,
    all_timeseries_data: dict[str, dict[str, pd.Series]],
    scenario: dict,
    heat_timeseries: dict[str, dict[str, pd.Series]] | None = None,
) -> pypsa.Network:
    """
    Create PyPSA network from scenario settings.
    """
    n = pypsa.Network()

    reference_country = scenario["countries"][0]
    n.set_snapshots(all_timeseries_data[reference_country]["load"].index)

    add_carriers(n)
    set_carrier_co2_emissions(n, cost_data)

    add_electricity(
        n=n,
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        scenario=scenario,
    )

    if scenario["with_battery_storage"]:
        add_battery_storage(
            n=n,
            cost_data=cost_data,
            scenario=scenario,
        )

    if scenario["with_interconnectors"]:
        add_interconnectors(n)

    if scenario["with_ch4_network"] or scenario["with_h2_network"]:
        add_gas(
            n=n,
            cost_data=cost_data,
            scenario=scenario,
        )

    if scenario["with_heat_sector"]:
        if heat_timeseries is None:
            raise ValueError(
                "Heat sector is enabled, but no heat_timeseries were provided."
            )

        add_heat(
            n=n,
            cost_data=cost_data,
            scenario=scenario,
            heat_timeseries=heat_timeseries,
        )

    if scenario["co2_limit"] is not None:
        add_global_co2_constraint(
            n=n,
            co2_limit=scenario["co2_limit"],
        )

    return n


# =========================================================
# SOLVING AND EXPORT
# =========================================================

def optimize_and_save_network(
    n: pypsa.Network,
    output_file: str | Path,
    scenario: dict,
    solver_name: str = "gurobi",
) -> None:
    """
    Optimize network and save it as NetCDF.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def extra_functionality(network: pypsa.Network, snapshots) -> None:
        custom_constraints(
            n=network,
            snapshots=snapshots,
            scenario=scenario,
        )

    n.optimize(
        n.snapshots,
        extra_functionality=extra_functionality,
        solver_name=solver_name,
        solver_options={},
        include_objective_constant=False,
    )

    n.export_to_netcdf(output_path)

    print(f"\nOptimized network saved to: {output_path}")


def print_model_summary(n: pypsa.Network) -> None:
    """
    Print compact model summary after optimization.
    """
    print("\nOptimized generator capacities [MW]:")
    if not n.generators.empty and "p_nom_opt" in n.generators.columns:
        print(n.generators[["carrier", "p_nom_opt"]])

    print("\nOptimized link capacities [MW]:")
    if not n.links.empty and "p_nom_opt" in n.links.columns:
        print(n.links[["carrier", "p_nom_opt"]])

    print("\nOptimized store capacities [MWh]:")
    if not n.stores.empty and "e_nom_opt" in n.stores.columns:
        print(n.stores[["carrier", "e_nom_opt"]])

    print("\nObjective value:")
    print(n.objective)


# =========================================================
# ENERGY BALANCE EXTRACTION
# =========================================================

def get_dk_electricity_balance_by_carrier(n: pypsa.Network) -> pd.DataFrame:
    """
    Return time-resolved DK electricity-bus balance by carrier.

    Positive values are injections into the DK electricity bus.
    Negative values are withdrawals from the DK electricity bus.
    """
    try:
        balance = n.statistics.energy_balance(
            aggregate_time=False,
            nice_names=False,
        )

        if "bus" in balance.index.names:
            balance_dk = balance.xs("DK", level="bus")
            balance_by_carrier = balance_dk.groupby(level="carrier").sum()
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
                ELECTRICITY_BALANCE_ORDER,
            )

            return balance_by_carrier_t

    except Exception as error:
        print(
            "Could not use n.statistics.energy_balance() for DK electricity balance. "
            f"Using component fallback. Reason: {error}"
        )

    balance_by_carrier_t = pd.DataFrame(index=n.snapshots)

    dk_loads = n.loads[n.loads.bus == "DK"].index
    if len(dk_loads) > 0:
        electricity_loads = [
            load for load in dk_loads if "electricity" in load or n.loads.at[load, "carrier"] == "AC"
        ]
        if electricity_loads:
            balance_by_carrier_t["electricity"] = -n.loads_t.p_set[
                electricity_loads
            ].sum(axis=1)

    dk_generators = n.generators[n.generators.bus == "DK"]

    for generator in dk_generators.index:
        carrier = map_carrier_to_display_name(n.generators.at[generator, "carrier"])

        if carrier not in balance_by_carrier_t.columns:
            balance_by_carrier_t[carrier] = 0.0

        balance_by_carrier_t[carrier] = balance_by_carrier_t[carrier].add(
            n.generators_t.p[generator].clip(lower=0.0),
            fill_value=0.0,
        )

    # Links withdrawing electricity from DK.
    for link in n.links.index[n.links.bus0 == "DK"]:
        carrier = map_carrier_to_display_name(n.links.at[link, "carrier"])

        if carrier not in ["battery charger", "electrolysis", "heat pump", "resistive heater"]:
            continue

        if carrier not in balance_by_carrier_t.columns:
            balance_by_carrier_t[carrier] = 0.0

        balance_by_carrier_t[carrier] = balance_by_carrier_t[carrier].add(
            -n.links_t.p0[link].clip(lower=0.0),
            fill_value=0.0,
        )

    # Links injecting electricity into DK.
    for link in n.links.index[n.links.bus1 == "DK"]:
        carrier = map_carrier_to_display_name(n.links.at[link, "carrier"])

        if carrier not in ["battery discharger", "CCGT", "H2 turbine", "H2 fuel cell"]:
            continue

        if carrier not in balance_by_carrier_t.columns:
            balance_by_carrier_t[carrier] = 0.0

        balance_by_carrier_t[carrier] = balance_by_carrier_t[carrier].add(
            (-n.links_t.p1[link]).clip(lower=0.0),
            fill_value=0.0,
        )

    balance_by_carrier_t = drop_empty_carriers(balance_by_carrier_t)
    balance_by_carrier_t = reorder_columns(
        balance_by_carrier_t,
        ELECTRICITY_BALANCE_ORDER,
    )

    return balance_by_carrier_t


def get_dk_heat_supply_by_carrier(n: pypsa.Network) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Return DK heat supply by carrier and DK heat demand.
    """
    heat_supply = pd.DataFrame(index=n.snapshots)
    heat_load = None

    if "DK_heat_demand" in n.loads_t.p_set.columns:
        heat_load = n.loads_t.p_set["DK_heat_demand"]

    for link in n.links.index[n.links.bus1 == "DK_heat"]:
        carrier = map_carrier_to_display_name(n.links.at[link, "carrier"])

        if carrier not in heat_supply.columns:
            heat_supply[carrier] = 0.0

        heat_supply[carrier] = heat_supply[carrier].add(
            (-n.links_t.p1[link]).clip(lower=0.0),
            fill_value=0.0,
        )

    heat_supply = drop_empty_carriers(heat_supply)
    heat_supply = reorder_columns(heat_supply, HEAT_BALANCE_ORDER)

    return heat_supply, heat_load


# =========================================================
# PLOTS
# =========================================================

def plot_denmark_dispatch_strategy(
    n: pypsa.Network,
    folder: Path,
) -> None:
    """
    Plot Denmark electricity dispatch, exchanges, and heat dispatch during the
    winter week with highest average Danish electricity demand.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    week_index = get_peak_winter_week_index(
        n=n,
        load_name="DK_electricity_demand",
    )

    electricity_balance = get_dk_electricity_balance_by_carrier(n).loc[week_index].copy()
    electricity_balance = drop_empty_carriers(electricity_balance)
    electricity_balance = reorder_columns(
        electricity_balance,
        ELECTRICITY_BALANCE_ORDER,
    )

    heat_supply, heat_load = get_dk_heat_supply_by_carrier(n)

    if not heat_supply.empty:
        heat_supply_week = heat_supply.loc[week_index].copy()
    else:
        heat_supply_week = pd.DataFrame(index=week_index)

    if heat_load is not None:
        heat_load_week = heat_load.loc[week_index]
    else:
        heat_load_week = None

    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]
    exchanges = {}

    for line in dk_lines.index:
        bus0 = n.lines.at[line, "bus0"]
        bus1 = n.lines.at[line, "bus1"]

        neighbour = bus1 if bus0 == "DK" else bus0
        flow = n.lines_t.p0[line].loc[week_index]

        # Positive = import to DK, negative = export from DK
        if bus0 == "DK":
            dk_exchange = -flow
        else:
            dk_exchange = flow

        exchanges[neighbour] = dk_exchange

    output_data = pd.DataFrame(index=week_index)

    for carrier in electricity_balance.columns:
        output_data[f"DK electricity balance {carrier} [MW]"] = electricity_balance[carrier]

    for neighbour, series in exchanges.items():
        output_data[f"DK exchange with {neighbour} [MW]"] = series

    if heat_load_week is not None:
        output_data["DK heat demand [MW]"] = heat_load_week

    for carrier in heat_supply_week.columns:
        output_data[f"DK heat supply {carrier} [MW]"] = heat_supply_week[carrier]

    output_data.to_csv(folder / "denmark_dispatch_strategy_winter_week.csv")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.5, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 2]},
    )

    ax1, ax2, ax3 = axes

    electricity_colors = get_carrier_colors(n, electricity_balance.columns)

    electricity_balance.plot.area(
        stacked=True,
        ax=ax1,
        color=electricity_colors,
        linewidth=0.0,
    )

    ax1.set_ylabel("Power balance [MW]")
    ax1.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    y_max = max(
        abs(float(electricity_balance.min().min())),
        abs(float(electricity_balance.max().max())),
    )
    y_max = int(np.ceil(y_max / 500.0) * 500.0)
    ax1.set_ylim(-y_max, y_max)

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )

    exchange_colors = {
        "DE": "#5AA469",
        "SE": "#D08770",
        "NO": "#2E86AB",
    }

    for neighbour, series in exchanges.items():
        ax2.plot(
            week_index,
            series.values,
            linewidth=2.0,
            label=neighbour,
            color=exchange_colors.get(neighbour, "#999999"),
        )

    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_ylabel("Exchange [MW]")
    ax2.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    if exchanges:
        ax2.legend(
            loc="upper right",
            ncol=3,
            frameon=True,
            framealpha=0.95,
        )

    if heat_load_week is not None and not heat_supply_week.empty:
        heat_colors = get_carrier_colors(n, heat_supply_week.columns)

        heat_supply_week.plot.area(
            stacked=True,
            ax=ax3,
            color=heat_colors,
            linewidth=0.0,
        )

        ax3.plot(
            week_index,
            heat_load_week,
            color="black",
            linewidth=2.2,
            label="heat demand",
        )

        ax3.legend(
            loc="upper right",
            ncol=2,
            frameon=True,
            framealpha=0.95,
        )

    else:
        ax3.text(
            0.5,
            0.5,
            "Heat sector not active for Denmark",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=REPORT_FONT_SIZE,
        )

    ax3.set_ylabel("Heat [MW]")
    ax3.set_xlabel("Time")
    ax3.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_axisbelow(True)

    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()

    save_figure(fig, folder / "denmark_dispatch_strategy_winter_week.png")
    plt.close(fig)


def plot_annual_mix_from_balance(
    n: pypsa.Network,
    output_dir: Path,
) -> None:
    """
    Plot annual Danish electricity mix from DK electricity balance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    balance = get_dk_electricity_balance_by_carrier(n)
    positive_balance = balance.clip(lower=0.0)

    annual_mix = positive_balance.sum(axis=0)
    annual_mix = annual_mix[annual_mix > 1e-3]
    annual_mix = annual_mix.reindex(
        [carrier for carrier in POSITIVE_ELECTRICITY_ORDER if carrier in annual_mix.index]
        + [carrier for carrier in annual_mix.index if carrier not in POSITIVE_ELECTRICITY_ORDER]
    )

    colors = get_carrier_colors(n, annual_mix.index)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    wedges, _, _ = ax.pie(
        annual_mix,
        colors=colors,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        textprops={"color": "black", "fontsize": REPORT_FONT_SIZE},
    )

    ax.legend(
        wedges,
        annual_mix.index,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        fontsize=REPORT_FONT_SIZE,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    save_figure(fig, output_dir / "annual_danish_electricity_mix.png")
    plt.close(fig)


def plot_capacity_factors_over_year(
    n: pypsa.Network,
    output_dir: Path,
) -> None:
    """
    Plot monthly capacity factors for selected Danish electricity technologies.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = n.snapshot_weightings.generators

    monthly_cf = {}

    generator_carriers = ["solar", "offwind", "coal", "nuclear", "gas CCGT"]

    gens = n.generators.index[
        (n.generators.bus == "DK")
        & (n.generators.carrier.isin(generator_carriers))
    ]

    for gen in gens:
        carrier = map_carrier_to_display_name(n.generators.at[gen, "carrier"])
        capacity = n.generators.at[gen, "p_nom_opt"]

        if capacity <= 1e-6:
            continue

        generation = n.generators_t.p[gen].clip(lower=0.0)
        monthly_generation = generation.multiply(weights, axis=0).resample("ME").sum()
        monthly_hours = weights.resample("ME").sum()

        monthly_cf[carrier] = monthly_generation / (capacity * monthly_hours)

    link_map = {
        "DK_CCGT": "CCGT",
        "DK_H2_turbine": "H2 turbine",
        "DK_H2_fuel_cell": "H2 fuel cell",
        "DK_battery_discharger": "battery discharger",
    }

    for link, label in link_map.items():
        if link not in n.links.index:
            continue

        capacity = n.links.at[link, "p_nom_opt"]

        if capacity <= 1e-6:
            continue

        output = (-n.links_t.p1[link]).clip(lower=0.0)

        monthly_output = output.multiply(weights, axis=0).resample("ME").sum()
        monthly_hours = weights.resample("ME").sum()

        monthly_cf[label] = monthly_output / (capacity * monthly_hours)

    cf = pd.DataFrame(monthly_cf)

    if cf.empty:
        print("No capacity factor data available.")
        return

    cf = cf.loc[:, cf.max() > 1e-6]
    cf = reorder_columns(cf, POSITIVE_ELECTRICITY_ORDER)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for column in cf.columns:
        ax.plot(
            cf.index,
            cf[column],
            label=column,
            color=get_carrier_color(n, column),
            linewidth=2.0,
        )

    ax.set_ylabel("Capacity factor [-]")
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.xaxis.get_offset_text().set_fontsize(REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, output_dir / "capacity_factors_over_year_dk.png")
    plt.close(fig)


# =========================================================
# RUN MODEL
# =========================================================

def main() -> None:
    set_report_plot_style()
    silence_gurobi_logger()

    scenario = SCENARIOS[ACTIVE_SCENARIO].copy()

    print(f"\nRunning scenario: {ACTIVE_SCENARIO}")
    print(f"Scenario name: {scenario['name']}")
    print(f"Weather year: {scenario['weather_year']}")
    print(f"Countries: {scenario['countries']}")

    cost_data = prepare_costs(
        cost_file=resolve_project_path(FILE_PATHS["cost_file"]),
        financial_parameters=FINANCIAL_PARAMETERS,
        number_of_years=FINANCIAL_PARAMETERS["nyears"],
    )

    all_timeseries_data = load_all_countries_timeseries(
        timeseries_file=resolve_project_path(FILE_PATHS["timeseries_file"]),
        countries=scenario["countries"],
        year=scenario["weather_year"],
    )

    heat_timeseries = None

    if scenario["with_heat_sector"]:
        heat_timeseries = load_heat_timeseries(
            heat_file=resolve_project_path(FILE_PATHS["heat_file"]),
            countries=scenario["countries"],
            year=scenario["weather_year"],
        )

    n = create_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        scenario=scenario,
        heat_timeseries=heat_timeseries,
    )

    output_file = (
        resolve_project_path(FILE_PATHS["network_output_dir"])
        / f"{scenario['name']}_{scenario['weather_year']}.nc"
    )

    optimize_and_save_network(
        n=n,
        output_file=output_file,
        scenario=scenario,
    )

    print_model_summary(n)

    plot_denmark_dispatch_strategy(
        n=n,
        folder=OUTPUT_DIR,
    )

    plot_annual_mix_from_balance(
        n=n,
        output_dir=OUTPUT_DIR,
    )

    plot_capacity_factors_over_year(
        n=n,
        output_dir=OUTPUT_DIR,
    )

    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()