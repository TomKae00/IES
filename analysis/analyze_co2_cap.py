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
from prettytable import PrettyTable
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pypsa

from model.helpers import (
    silence_gurobi_logger,
    prepare_costs,
    load_all_countries_timeseries,
    calculate_conventional_marginal_cost
)

from model.scenarios import (
    FINANCIAL_PARAMETERS,
    FILE_PATHS,
    SCENARIOS,
)


# Change this to run another scenario
ACTIVE_SCENARIO = "co2_cap"


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

            ch4_capital_cost = cost_data.at[ch4_tech_name, "fixed"] * length_km
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

            h2_capital_cost = cost_data.at[h2_tech_name, "fixed"] * length_km
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
# CO2 Cap extension
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
    

def calculate_total_emissions(n: pypsa.Network) -> float:
    """
    Calculate total CO2 emissions [tCO2] from generator dispatch.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.

    Returns
    -------
    float
        Total CO2 emissions in tCO2.
    """
    total_emissions = 0.0

    for generator in n.generators.index:
        carrier = n.generators.at[generator, "carrier"]

        if carrier not in n.carriers.index or "co2_emissions" not in n.carriers.columns:
            continue

        co2_intensity = n.carriers.at[carrier, "co2_emissions"]

        if pd.isna(co2_intensity) or co2_intensity == 0:
            continue

        efficiency = (
            n.generators.at[generator, "efficiency"]
            if pd.notna(n.generators.at[generator, "efficiency"])
            else 1.0
        )

        dispatch = n.generators_t.p[generator].sum()  # MWh_el
        emissions = dispatch / efficiency * co2_intensity  # tCO2

        total_emissions += emissions

    return total_emissions


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


def custom_constraints(n: pypsa.Network, snapshots) -> None:
    """
    Add custom optimization constraints.
    """
    add_battery_charger_ratio_constraints(n)
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
    solver_name: str = "gurobi",
) -> None:
    """
    Optimize network and save it as NetCDF.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n.optimize(
        n.snapshots,
        extra_functionality=custom_constraints,
        solver_name=solver_name,
        solver_options={},
    )

    n.export_to_netcdf(output_path)

    print(f"\nOptimized network saved to: {output_path}")


def print_model_summary(n: pypsa.Network, cap: float) -> None:
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

    print("\nGlobal constraints:")
    print(n.global_constraints)

    co2_constraint_name = "CO2_limit"

    if ("mu" in n.global_constraints.columns and co2_constraint_name in n.global_constraints.index):
        co2_price = abs(n.global_constraints.loc[co2_constraint_name, "mu"])
        print("\nImplied CO2 price [EUR/tCO2]:")
        print(co2_price)
    else:
        co2_price = None
        print("\nImplied CO2 price [EUR/tCO2]:")
        print(co2_price)
        
    total_emissions = calculate_total_emissions(n)

    print("\nTotal CO2 emissions [tCO2]:")
    print(total_emissions)

    print("\nTotal emission acceptance and co2 cap price for different scenarios [tCO2]:")
    print(" ")
    t = PrettyTable(["Allowed Emissions Share", "Total Emission Acceptance [tCO2]", "CO2 Cap Price [EUR/tCO2]"])
    t.add_row([
        cap if cap is not None else "No cap",
        total_emissions,
        co2_price if co2_price is not None else "No CO2 constraint",
    ])
    print(t)

# =========================================================
# RUNNING THE MODEL
# =========================================================

def main() -> None:
    silence_gurobi_logger()

    FIND_BASELINE = False  # Set to False to run the CO2 cap scenario instead of the baseline

    scenario = SCENARIOS[ACTIVE_SCENARIO].copy()

    if FIND_BASELINE:
        scenario["co2_price"] = 0.0
        scenario["co2_limit"] = None
    else:
        baseline_emissions = 603779863.8428456
        scenario["co2_limit"] = baseline_emissions * scenario["co2_cap"]

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

    n = create_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        scenario=scenario,
        heat_timeseries=None,
    )

    run_type = "baseline" if FIND_BASELINE else "cap"

    output_file = (
        Path(FILE_PATHS["network_output_dir"])
        / f"{scenario['name']}_{run_type}_{scenario['weather_year']}.nc"
    )

    optimize_and_save_network(n=n, output_file=output_file)

    print_model_summary(
        n,
        cap=None if FIND_BASELINE else scenario["co2_cap"],
    )


if __name__ == "__main__":
    main()