"""
PyPSA model construction for the IEG course project.

This file contains only the model structure:
- carriers
- buses, loads, generators
- storage
- electricity interconnectors
- gas network
- heat sector coupling
- CO2 constraint
- optimization helper

Scenario definitions are stored in scenarios.py.
Data loading and cost preparation are stored in helpers.py.
The selected scenario is run from run_model.py.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import numpy as np


from model.helpers import (
    silence_gurobi_logger,
    prepare_costs,
    load_all_countries_timeseries,
    load_heat_timeseries,
    calculate_conventional_marginal_cost,
)

from model.scenarios import (
    FINANCIAL_PARAMETERS,
    FILE_PATHS,
    SCENARIOS,
)


# Change this to run another scenario
ACTIVE_SCENARIO = "sector_coupling"


# =========================================================
# CARRIERS
# =========================================================

def add_carriers(n: pypsa.Network) -> None:
    """
    Add all carriers used across possible scenarios.
    """
    carrier_data = {
        # Electricity
        "AC": {"color": "#4C566A"},

        # Electricity generation
        "solar": {"color": "#EBCB3B"},
        "onwind": {"color": "#5AA469"},
        "offwind": {"color": "#2E86AB"},
        "gas CCGT": {"color": "#D08770"},
        "coal": {"color": "#5C5C5C"},
        "nuclear": {"color": "#8F6BB3"},

        # Battery
        "battery": {"color": "#E67E22"},
        "battery charger": {"color": "#C06C84"},
        "battery discharger": {"color": "#6C5B7B"},

        # CH4 network
        "CH4": {"color": "#A35D3D"},
        "CH4 supply": {"color": "#8C564B"},
        "CH4 pipeline": {"color": "#7B4B2A"},
        "CCGT": {"color": "#D08770"},

        # H2 network
        "H2": {"color": "#4DA3FF"},
        "H2 store": {"color": "#7FDBFF"},
        "H2 pipeline": {"color": "#2F6DB3"},
        "electrolysis": {"color": "#3A86FF"},
        "H2 turbine": {"color": "#6FA8DC"},
        "H2 fuel cell": {"color": "#5E81AC"},

        # Heat sector
        "heat": {"color": "#B48EAD"},
        "heat pump": {"color": "#A3BE8C"},
        "resistive heater": {"color": "#BF616A"},
        "gas boiler": {"color": "#A35D3D"},

        # Heat storage
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

    Values are assumed to be in tCO2/MWh_fuel if available in cost_data.
    """
    if "co2_emissions" not in n.carriers.columns:
        n.carriers["co2_emissions"] = 0.0

    if "gas" in cost_data.index and "CO2 intensity" in cost_data.columns:
        gas_co2_intensity = cost_data.at["gas", "CO2 intensity"]

        # Simplified gas-fired electricity generator
        n.carriers.loc["gas CCGT", "co2_emissions"] = gas_co2_intensity

        # Explicit CH4 supply into the CH4 network
        n.carriers.loc["CH4 supply", "co2_emissions"] = gas_co2_intensity

        # Simplified gas boiler generator in the heat sector
        n.carriers.loc["gas boiler", "co2_emissions"] = gas_co2_intensity

    if "coal" in cost_data.index and "CO2 intensity" in cost_data.columns:
        n.carriers.loc["coal", "co2_emissions"] = cost_data.at[
            "coal", "CO2 intensity"
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

    If the gas network is enabled, CCGT is not added here as a normal generator.
    Instead, it is added later in add_gas() as a CH4-to-electricity link.
    """
    for country_code in scenario["countries"]:
        timeseries_data = all_timeseries_data[country_code]

        # -----------------------------
        # Electricity bus and demand
        # -----------------------------
        n.add(
            "Bus",
            country_code,
            carrier="AC",
        )

        n.add(
            "Load",
            f"{country_code}_electricity_demand",
            bus=country_code,
            carrier="AC",
            p_set=timeseries_data["load"],
        )

        # -----------------------------
        # Solar PV
        # -----------------------------
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

        # -----------------------------
        # Onshore wind
        # -----------------------------
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

        # -----------------------------
        # Offshore wind
        # -----------------------------
        if country_code == "DK":
            n.add(
                "Generator",
                f"{country_code}_offshore_wind",
                bus=country_code,
                carrier="offwind",
                p_nom_extendable=True,
                p_nom_max=2650,
                p_max_pu=timeseries_data["offshore_wind_cf"],
                capital_cost=cost_data.at["offwind", "fixed"],
                marginal_cost=cost_data.at["offwind", "VOM"],
                )
        else:
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

        # -----------------------------
        # CCGT
        # -----------------------------
        # Only add CCGT as a normal generator if gas is not explicitly modeled.
        # If with_gas_network=True, CCGT is added later as a CH4-to-electricity link.
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

        # -----------------------------
        # Coal
        # -----------------------------
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

        # -----------------------------
        # Nuclear
        # -----------------------------
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

                # ensure nuclear is run less flexible:
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

    Each battery is represented by:
    - one battery bus
    - one Store for energy capacity
    - one charging Link from the electricity bus to the battery bus
    - one discharging Link from the battery bus to the electricity bus
    """
    for country_code in scenario["countries"]:
        battery_bus = f"{country_code}_battery"

        # -----------------------------
        # Battery bus
        # -----------------------------
        n.add(
            "Bus",
            battery_bus,
            carrier="battery",
        )

        # -----------------------------
        # Battery energy capacity
        # -----------------------------
        n.add(
            "Store",
            f"{country_code}_battery_store",
            bus=battery_bus,
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=cost_data.at["battery storage", "fixed"],
        )

        # -----------------------------
        # Battery charging link
        # -----------------------------
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

        # -----------------------------
        # Battery discharging link
        # -----------------------------
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
    Add fixed electricity interconnectors between the modeled countries.

    Lines are bidirectional in PyPSA. bus0 and bus1 only define the positive
    flow direction.
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
            #v_nom=400.0,
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
    Add gas sector components to the network.

    The gas sector can include CH4, H2, or both, depending on the scenario:

    - with_ch4_network:
        CH4 buses, CH4 supply, CCGT links, CH4 pipelines

    - with_h2_network:
        H2 buses, electrolysis, optional H2 turbine/fuel cell, H2 pipelines

    Note
    ----
    If the CH4 network is enabled, CCGT should not be added as a normal
    electricity generator in add_electricity(). Instead, it is represented here
    as a conversion link from CH4 to electricity.
    """
    countries = scenario["countries"]

    with_ch4_network = scenario.get("with_ch4_network", True)
    with_h2_network = scenario.get("with_h2_network", True)

    if not with_ch4_network and not with_h2_network:
        raise ValueError("add_gas() was called although both CH4 and H2 networks are disabled.")

    for country_code in countries:
        # -----------------------------
        # CH4 system
        # -----------------------------
        if with_ch4_network:
            n.add(
                "Bus",
                f"{country_code}_CH4",
                carrier="CH4",
            )

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

        # -----------------------------
        # H2 system
        # -----------------------------
        if with_h2_network:
            n.add(
                "Bus",
                f"{country_code}_H2",
                carrier="H2",
            )

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

            # H2 storage
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

    # -----------------------------
    # CH4 supply
    # -----------------------------
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

    # -----------------------------
    # Gas pipelines
    # -----------------------------
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

        # -----------------------------
        # CH4 pipeline
        # -----------------------------
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
                #capital_cost=ch4_capital_cost,
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
                #capital_cost=ch4_capital_cost,
                marginal_cost=0.0,
            )

        # -----------------------------
        # H2 pipeline
        # -----------------------------
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
                #capital_cost=h2_capital_cost,
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
                #capital_cost=h2_capital_cost,
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

    The heat sector includes:
    - one heat bus per country
    - one heat demand per country
    - one decentral air-sourced heat pump per country
    - one decentral resistive heater per country
    - one decentral gas boiler per country
    - optionally one decentral water tank storage per country

    Heat demand:
        {country_code}_heat_demand_total

    Heat pump COP:
        {country_code}_COP_ASHP_floor

    For Norway, Danish heat demand and COP profiles are used as a proxy in
    helpers.load_heat_timeseries().
    """
    with_ch4_network = scenario.get("with_ch4_network", False)

    for country_code in scenario["countries"]:
        if country_code not in heat_timeseries:
            raise KeyError(f"No heat time series found for {country_code}.")

        heat_demand = heat_timeseries[country_code]["heat_demand"]
        heat_pump_cop = heat_timeseries[country_code]["ashp_cop"]

        heat_bus = f"{country_code}_heat"
        heat_storage_bus = f"{country_code}_heat_storage"

        # -----------------------------
        # Heat bus
        # -----------------------------
        n.add(
            "Bus",
            heat_bus,
            carrier="heat",
        )

        # -----------------------------
        # Heat demand
        # -----------------------------
        n.add(
            "Load",
            f"{country_code}_heat_demand",
            bus=heat_bus,
            carrier="heat",
            p_set=heat_demand,
        )

        # -----------------------------
        # Decentral air-sourced heat pump: electricity -> heat
        # -----------------------------
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

        # -----------------------------
        # Decentral resistive heater: electricity -> heat
        # -----------------------------
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

        # -----------------------------
        # Decentral gas boiler
        # -----------------------------
        gas_boiler_tech = "decentral gas boiler"
        gas_boiler_efficiency = cost_data.at[gas_boiler_tech, "efficiency"]

        if with_ch4_network:
            # If CH4 is explicitly modeled, the gas boiler consumes CH4 from
            # the country CH4 bus. Fuel cost and emissions are accounted for
            # when CH4 enters the system through CH4 supply.
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
            # If CH4 is not explicitly modeled, represent the gas boiler as a
            # fuel-consuming heat generator. In this case, the CO2 constraint
            # can account for emissions through the gas boiler carrier.
            gas_fuel_cost = cost_data.at["gas", "fuel"]

            if "CO2 intensity" in cost_data.columns:
                gas_co2_intensity = cost_data.at["gas", "CO2 intensity"]
                gas_boiler_co2_cost = (
                    gas_co2_intensity
                    / gas_boiler_efficiency
                    * scenario["co2_price"]
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

        # -----------------------------
        # Optional decentral water tank storage
        # -----------------------------
        if scenario.get("with_heat_storage", False):
            water_tank_storage_tech = "decentral water tank storage"
            water_tank_charger_tech = "decentral water tank charger"
            water_tank_discharger_tech = "decentral water tank discharger"

            n.add(
                "Bus",
                heat_storage_bus,
                carrier="heat storage",
            )

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

            # Store the energy-to-power ratio as metadata. This does not yet
            # enforce the ratio as a constraint.
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
                ]
                / 100,
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

    co2_limit is the allowed total emissions over the modeled period in tCO2.
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

    For each battery, enforce:

        charger_p_nom - discharger_efficiency * discharger_p_nom == 0
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

    For each water tank storage unit, enforce:

        Store-e_nom - energy_to_power_ratio * Link-p_nom == 0
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

    def extra_functionality(n: pypsa.Network, snapshots) -> None:
        custom_constraints(
            n=n,
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
    if not n.generators.empty:
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
# PLOTS
# =========================================================

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

def plot_denmark_dispatch_strategy(n, folder):
    """
    Plot Denmark dispatch during the winter week with highest average
    Danish electricity demand.

    Includes:
    - electricity generation
    - battery charge/discharge
    - gas-to-power and H2-to-power links
    - electricity exchanges
    - heat demand and heat supply
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    snapshots = n.snapshots

    winter_mask = (snapshots.month == 12) | (snapshots.month == 1)

    dk_load = n.loads_t.p_set["DK_electricity_demand"]
    winter_load = dk_load[winter_mask]

    weekly_avg_load = winter_load.resample("W").mean()
    max_load_week_end = weekly_avg_load.idxmax()
    week_start = max_load_week_end - pd.Timedelta(days=6)
    week_end = max_load_week_end

    week_index = dk_load.loc[week_start:week_end].index
    week_load = dk_load.loc[week_index]

    # -------------------------------------------------
    # Electricity supply at DK bus
    # -------------------------------------------------
    electricity_supply = {}

    # Generators directly connected to DK
    dk_generators = n.generators[n.generators.bus == "DK"]

    for gen in dk_generators.index:
        carrier = n.generators.at[gen, "carrier"]

        electricity_supply.setdefault(
            carrier, pd.Series(0.0, index=week_index)
        )

        electricity_supply[carrier] += n.generators_t.p[gen].loc[week_index]

    # Links producing electricity into DK
    link_supply_carriers = {
        "DK_battery_discharger": "battery discharge",
        "DK_CCGT": "gas CCGT",
        "DK_H2_turbine": "H2 turbine",
        "DK_H2_fuel_cell": "H2 fuel cell",
    }

    for link, carrier in link_supply_carriers.items():
        if link in n.links.index and link in n.links_t.p1.columns:
            electricity_supply.setdefault(
                carrier, pd.Series(0.0, index=week_index)
            )

            # p1 is negative when power is delivered to bus1
            electricity_supply[carrier] += (
                -n.links_t.p1[link].loc[week_index]
            ).clip(lower=0)

    # Electricity consumption by links at DK bus
    electricity_consumption = {}

    link_consumption_carriers = {
        "DK_battery_charger": "battery charge",
        "DK_electrolyzer": "electrolysis",
        "DK_ASHP": "heat pump electricity",
        "DK_resistive_heater": "resistive heater electricity",
    }

    for link, carrier in link_consumption_carriers.items():
        if link in n.links.index and link in n.links_t.p0.columns:
            electricity_consumption[carrier] = (
                n.links_t.p0[link].loc[week_index]
            ).clip(lower=0)

    # -------------------------------------------------
    # Electricity exchanges
    # -------------------------------------------------
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

        exchanges.setdefault(neighbour, pd.Series(0.0, index=week_index))
        exchanges[neighbour] += dk_exchange

    # -------------------------------------------------
    # Heat sector
    # -------------------------------------------------
    heat_supply = {}
    heat_load = None

    if "DK_heat_demand" in n.loads.index:
        heat_load = n.loads_t.p_set["DK_heat_demand"].loc[week_index]

    heat_links = {
        "DK_ASHP": "heat pump",
        "DK_resistive_heater": "resistive heater",
        "DK_gas_boiler": "gas boiler",
        "DK_water_tank_discharger": "heat storage discharge",
    }

    for link, carrier in heat_links.items():
        if link in n.links.index and link in n.links_t.p1.columns:
            heat_supply[carrier] = (
                -n.links_t.p1[link].loc[week_index]
            ).clip(lower=0)

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    fig, axes = plt.subplots(
        3, 1,
        figsize=(15, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3, 2]},
    )

    ax1, ax2, ax3 = axes

    colors = {
        "solar": "#ffd92f",
        "onwind": "#1b9e77",
        "offwind": "#377eb8",
        "gas CCGT": "#e41a1c",
        "CCGT": "#e41a1c",
        "coal": "#4d4d4d",
        "nuclear": "#984ea3",
        "battery discharge": "#ff7f00",
        "H2 turbine": "#6FA8DC",
        "H2 fuel cell": "#5E81AC",
        "heat pump": "#A3BE8C",
        "resistive heater": "#BF616A",
        "gas boiler": "#A35D3D",
        "heat storage discharge": "#81A1C1",
    }

    electricity_order = [
        "solar",
        "onwind",
        "offwind",
        "gas CCGT",
        "CCGT",
        "coal",
        "nuclear",
        "battery discharge",
        "H2 turbine",
        "H2 fuel cell",
    ]

    electricity_order += [
        c for c in electricity_supply if c not in electricity_order
    ]

    bottom = np.zeros(len(week_index))

    for carrier in electricity_order:
        if carrier not in electricity_supply:
            continue

        series = electricity_supply[carrier].fillna(0).values

        ax1.fill_between(
            week_index,
            bottom,
            bottom + series,
            label=f"DK {carrier}",
            color=colors.get(carrier),
            alpha=0.85,
        )

        bottom += series

    ax1.plot(
        week_index,
        week_load,
        color="black",
        linewidth=2.3,
        label="DK electricity demand",
    )

    for carrier, series in electricity_consumption.items():
        ax1.plot(
            week_index,
            -series,
            linestyle="--",
            linewidth=1.8,
            label=carrier,
        )

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Power [MW]")
    ax1.set_title("Denmark electricity dispatch during peak winter week")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))

    # Exchanges
    exchange_colors = {
        "DE": "#2ca02c",
        "SE": "#ff7f0e",
        "NO": "#1f77b4",
    }

    for neighbour, series in exchanges.items():
        ax2.plot(
            week_index,
            series,
            linewidth=2.2,
            label=f"{neighbour}",
            color=exchange_colors.get(neighbour),
        )

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Import / export [MW]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="Exchange", loc="upper left", bbox_to_anchor=(1.01, 1))

    # Heat
    if heat_load is not None and heat_supply:
        heat_order = [
            "heat pump",
            "resistive heater",
            "gas boiler",
            "heat storage discharge",
        ]

        bottom = np.zeros(len(week_index))

        for carrier in heat_order:
            if carrier not in heat_supply:
                continue

            series = heat_supply[carrier].fillna(0).values

            ax3.fill_between(
                week_index,
                bottom,
                bottom + series,
                label=f"DK {carrier}",
                color=colors.get(carrier),
                alpha=0.85,
            )

            bottom += series

        ax3.plot(
            week_index,
            heat_load,
            color="black",
            linewidth=2.3,
            label="DK heat demand",
        )

        ax3.set_ylabel("Heat [MW]")
        ax3.set_title("Denmark heat dispatch")
        ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        ax3.text(
            0.5,
            0.5,
            "Heat sector not active for Denmark",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_ylabel("Heat [MW]")

    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Time")

    plt.xticks(rotation=45)
    plt.tight_layout()

    outfile = folder / "denmark_dispatch_strategy_winter_week.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_annual_mix_from_balance(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot annual Danish electricity mix only.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annual_mix = {}

    # -----------------------------
    # 1. DK generators
    # -----------------------------
    dk_generators = n.generators[n.generators.bus == "DK"]

    for gen in dk_generators.index:
        carrier = n.generators.at[gen, "carrier"]

        generation = n.generators_t.p[gen].clip(lower=0).sum()

        annual_mix[carrier] = annual_mix.get(carrier, 0.0) + generation

    # -----------------------------
    # 2. Links producing electricity into DK
    # -----------------------------
    dk_electricity_links = n.links[n.links.bus1 == "DK"]

    for link in dk_electricity_links.index:
        carrier = n.links.at[link, "carrier"]

        output_to_dk = (-n.links_t.p1[link]).clip(lower=0).sum()

        annual_mix[carrier] = annual_mix.get(carrier, 0.0) + output_to_dk

    annual_mix = pd.Series(annual_mix)

    # -----------------------------
    # Remove very small numerical values
    # -----------------------------
    tolerance = 1e-3
    annual_mix = annual_mix[annual_mix > tolerance]

    # -----------------------------
    # Optional: manually exclude onshore wind
    # -----------------------------
    annual_mix = annual_mix.drop("onwind", errors="ignore")

    # -----------------------------
    # Get colors BEFORE renaming
    # -----------------------------
    colors = get_carrier_colors(n, annual_mix.index)

    # -----------------------------
    # Nice labels for legend
    # -----------------------------
    label_map = {
        "solar": "solar PV",
        "onwind": "onshore wind",
        "offwind": "offshore wind",
        "gas CCGT": "gas CCGT",
        "CCGT": "gas CCGT",
        "coal": "coal",
        "nuclear": "nuclear",
        "battery discharger": "battery discharge",
        "H2 turbine": "H2 turbine",
        "H2 fuel cell": "H2 fuel cell",
    }

    labels = [label_map.get(carrier, carrier) for carrier in annual_mix.index]

    fig, ax = plt.subplots(figsize=(6, 5))

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
        labels,
        title="Technology",
        loc="lower right",
        bbox_to_anchor=(1, 0),
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "annual_danish_electricity_mix.png", dpi=300)
    plt.close(fig)


def plot_capacity_factors_over_year(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot monthly capacity factors for selected Danish electricity technologies.
    Includes both generators and gas/H2/battery links producing electricity into DK.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = n.snapshot_weightings.generators

    monthly_cf = {}

    # -----------------------------
    # 1. Danish generators
    # -----------------------------
    generator_carriers = ["solar", "offwind", "coal", "nuclear", "gas CCGT"]

    gens = n.generators.index[
        (n.generators.bus == "DK")
        & (n.generators.carrier.isin(generator_carriers))
    ]

    for gen in gens:
        carrier = n.generators.at[gen, "carrier"]
        capacity = n.generators.at[gen, "p_nom_opt"]

        if capacity <= 1e-6:
            continue

        generation = n.generators_t.p[gen].clip(lower=0)
        monthly_generation = generation.multiply(weights, axis=0).resample("ME").sum()
        monthly_hours = weights.resample("ME").sum()

        cf = monthly_generation / (capacity * monthly_hours)
        monthly_cf[carrier] = cf

    # -----------------------------
    # 2. Danish electricity-producing links
    # -----------------------------
    link_map = {
        "DK_CCGT": "gas CCGT",
        "DK_H2_turbine": "H2 turbine",
        "DK_H2_fuel_cell": "H2 fuel cell",
        "DK_battery_discharger": "battery discharge",
    }

    for link, label in link_map.items():
        if link not in n.links.index:
            continue

        capacity = n.links.at[link, "p_nom_opt"]

        if capacity <= 1e-6:
            continue

        # p1 is negative when output is delivered to DK
        output = (-n.links_t.p1[link]).clip(lower=0)

        monthly_output = output.multiply(weights, axis=0).resample("ME").sum()
        monthly_hours = weights.resample("ME").sum()

        cf = monthly_output / (capacity * monthly_hours)
        monthly_cf[label] = cf

    cf = pd.DataFrame(monthly_cf)

    cf = cf.loc[:, cf.max() > 1e-6]

    color_keys = {
        "solar": "solar",
        "offwind": "offwind",
        "coal": "coal",
        "nuclear": "nuclear",
        "gas CCGT": "CCGT",
        "H2 turbine": "H2 turbine",
        "H2 fuel cell": "H2 fuel cell",
        "battery discharge": "battery discharger",
    }

    label_map = {
        "solar": "solar PV",
        "offwind": "offshore wind",
        "coal": "coal",
        "nuclear": "nuclear",
        "gas CCGT": "gas CCGT",
        "H2 turbine": "H2 turbine",
        "H2 fuel cell": "H2 fuel cell",
        "battery discharge": "battery discharge",
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    for column in cf.columns:
        carrier_key = color_keys.get(column, column)

        if carrier_key in n.carriers.index:
            color = n.carriers.at[carrier_key, "color"]
        else:
            color = "#999999"

        ax.plot(
            cf.index,
            cf[column],
            label=label_map.get(column, column),
            color=color,
            linewidth=2,
        )

    ax.set_ylabel("Capacity factor [-]")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Technology")

    fig.tight_layout()
    fig.savefig(output_dir / "capacity_factors_over_year_dk.png", dpi=300)
    plt.close(fig)


# =========================================================
# RUN MODEL
# =========================================================


def main() -> None:
    silence_gurobi_logger()

    scenario = SCENARIOS[ACTIVE_SCENARIO]

    print(f"\nRunning scenario: {ACTIVE_SCENARIO}")
    print(f"Scenario name: {scenario['name']}")
    print(f"Weather year: {scenario['weather_year']}")
    print(f"Countries: {scenario['countries']}")

    cost_data = prepare_costs(
        cost_file=FILE_PATHS["cost_file"],
        financial_parameters=FINANCIAL_PARAMETERS,
        number_of_years=FINANCIAL_PARAMETERS["nyears"],
    )

    all_timeseries_data = load_all_countries_timeseries(
        timeseries_file=FILE_PATHS["timeseries_file"],
        countries=scenario["countries"],
        year=scenario["weather_year"],
    )

    heat_timeseries = None

    if scenario["with_heat_sector"]:
        heat_timeseries = load_heat_timeseries(
            heat_file=FILE_PATHS["heat_file"],
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
            Path(FILE_PATHS["network_output_dir"])
            / f"{scenario['name']}_{scenario['weather_year']}.nc"
    )

    optimize_and_save_network(
        n=n,
        output_file=output_file,
        scenario=scenario
    )

    print_model_summary(n)

    plot_denmark_dispatch_strategy(
        n=n,
        folder=Path("results/experiments")
    )
    plot_annual_mix_from_balance(
        n=n,
        output_dir=Path("results/experiments"),
    )

    plot_capacity_factors_over_year(
        n=n,
        output_dir=Path("results/experiments"),
    )

if __name__ == "__main__":
    main()