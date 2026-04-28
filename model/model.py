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

import pandas as pd
import pypsa

from model.helpers import calculate_conventional_marginal_cost


# =========================================================
# CARRIERS
# =========================================================

def add_carriers(n: pypsa.Network) -> None:
    """
    Add all carriers used across possible scenarios.
    """
    carrier_data = {
        "electricity": {"color": "#4C566A"},
        "solar": {"color": "#EBCB3B"},
        "onwind": {"color": "#5AA469"},
        "offwind": {"color": "#2E86AB"},
        "gas": {"color": "#D08770"},
        "coal": {"color": "#5C5C5C"},
        "nuclear": {"color": "#8F6BB3"},
        "battery": {"color": "#E67E22"},
        "battery charger": {"color": "#C06C84"},
        "battery discharger": {"color": "#6C5B7B"},
        "CH4": {"color": "#A35D3D"},
        "H2": {"color": "#4DA3FF"},
        "CH4 pipeline": {"color": "#7B4B2A"},
        "H2 pipeline": {"color": "#2F6DB3"},
        "CH4 supply": {"color": "#8C564B"},
        "electrolysis": {"color": "#3A86FF"},
        "CCGT": {"color": "#D08770"},
        "H2 turbine": {"color": "#6FA8DC"},
        "heat": {"color": "#B48EAD"},
        "heat pump": {"color": "#A3BE8C"},
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
        n.carriers.loc["gas", "co2_emissions"] = cost_data.at[
            "gas", "CO2 intensity"
        ]
        n.carriers.loc["CH4 supply", "co2_emissions"] = cost_data.at[
            "gas", "CO2 intensity"
        ]

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
            carrier="electricity",
        )

        n.add(
            "Load",
            f"{country_code}_electricity_demand",
            bus=country_code,
            carrier="electricity",
            p_set=timeseries_data["load"],
        )

        # -----------------------------
        # Solar PV
        # -----------------------------
        if "solar" in cost_data.index:
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
        if "onwind" in cost_data.index:
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
        if "offwind" in cost_data.index:
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
        if not scenario["with_gas_network"] and "CCGT" in cost_data.index:
            marginal_cost = calculate_conventional_marginal_cost(
                cost_data=cost_data,
                technology="CCGT",
                co2_price=scenario["co2_price"],
            )

            n.add(
                "Generator",
                f"{country_code}_CCGT",
                bus=country_code,
                carrier="gas",
                p_nom_extendable=True,
                capital_cost=cost_data.at["CCGT", "fixed"],
                marginal_cost=marginal_cost,
                efficiency=cost_data.at["CCGT", "efficiency"],
            )

        # -----------------------------
        # Coal
        # -----------------------------
        if "coal" in cost_data.index:
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

                # Optional if you want less flexible nuclear:
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
            r=0.0,
            v_nom=400.0,
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

    This includes:
    - CH4 and H2 buses for each modeled country
    - CCGT as CH4-to-electricity conversion
    - electrolysis as electricity-to-H2 conversion
    - optional H2 turbine as H2-to-electricity conversion
    - CH4 supply in Norway
    - CH4 and H2 pipelines between modeled countries

    Note
    ----
    If the gas network is enabled, CCGT should not be added as a normal
    electricity generator in add_electricity(). Instead, it is represented here
    as a conversion link from CH4 to electricity.
    """
    countries = scenario["countries"]

    for country_code in countries:
        # -----------------------------
        # Gas buses
        # -----------------------------
        n.add(
            "Bus",
            f"{country_code}_CH4",
            carrier="CH4",
        )

        n.add(
            "Bus",
            f"{country_code}_H2",
            carrier="H2",
        )

        # -----------------------------
        # CCGT: CH4 -> electricity
        # -----------------------------
        if "CCGT" in cost_data.index:
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
        # Electrolysis: electricity -> H2
        # -----------------------------
        if "electrolysis" in cost_data.index:
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

        # -----------------------------
        # Optional H2 turbine: H2 -> electricity
        # -----------------------------
        if scenario.get("with_h2_turbine", False) and "OCGT" in cost_data.index:
            n.add(
                "Link",
                f"{country_code}_H2_turbine",
                bus0=f"{country_code}_H2",
                bus1=country_code,
                carrier="H2 turbine",
                p_nom_extendable=True,
                efficiency=0.5,
                capital_cost=cost_data.at["OCGT", "fixed"],
                marginal_cost=cost_data.at["OCGT", "VOM"],
            )

    # -----------------------------
    # CH4 supply
    # -----------------------------
    if "NO" in countries:
        if "gas" not in cost_data.index:
            raise KeyError("'gas' must exist in cost_data to model CH4 supply.")

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
        ("NO", "DK", "NO_DK", 600.0, True),
        ("NO", "SE", "NO_SE", 500.0, True),
        ("NO", "DE", "NO_DE", 900.0, True),
        ("DK", "SE", "DK_SE", 300.0, True),
        ("DK", "DE", "DK_DE", 250.0, True),
        ("DE", "SE", "DE_SE", 700.0, True),
    ]

    for country_a, country_b, corridor_name, length_km, submarine in gas_corridor_data:
        if country_a not in countries or country_b not in countries:
            print(
                f"Skipping gas corridor {corridor_name}: "
                f"{country_a} or {country_b} not in network."
            )
            continue

        # -----------------------------
        # CH4 pipeline
        # -----------------------------
        if submarine:
            ch4_tech_name = "CH4 (g) submarine pipeline"
        else:
            ch4_tech_name = "CH4 (g) pipeline"

        if ch4_tech_name in cost_data.index:
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
                capital_cost=ch4_capital_cost,
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
                capital_cost=ch4_capital_cost,
                marginal_cost=0.0,
            )

        else:
            print(f"Warning: {ch4_tech_name} not found in cost_data. Skipping CH4 pipeline.")

        # -----------------------------
        # H2 pipeline
        # -----------------------------
        if submarine:
            h2_tech_name = "H2 (g) submarine pipeline"
        else:
            h2_tech_name = "H2 (g) pipeline"

        if h2_tech_name in cost_data.index:
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
                capital_cost=h2_capital_cost,
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
                capital_cost=h2_capital_cost,
                marginal_cost=0.0,
            )

        else:
            print(f"Warning: {h2_tech_name} not found in cost_data. Skipping H2 pipeline.")

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
    Add a simple heat sector to all modeled countries.

    The heat sector is represented by:
    - one heat bus per country
    - one heat demand per country
    - one air-source heat pump converting electricity to heat per country
    - optionally one heat storage per country

    Heat demand:
        {country_code}_heat_demand_total

    Heat pump COP:
        {country_code}_COP_ASHP_floor

    For Norway, Danish heat demand and COP profiles are used as a proxy in
    helpers.load_heat_timeseries().
    """
    for country_code in scenario["countries"]:
        if country_code not in heat_timeseries:
            raise KeyError(
                f"No heat time series found for {country_code}."
            )

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
        # Air-source heat pump: electricity -> heat
        # -----------------------------
        possible_heat_pump_cost_names = [
            "central air-sourced heat pump",
            "decentral air-sourced heat pump",
            "air-sourced heat pump",
        ]

        heat_pump_cost_name = next(
            (
                technology
                for technology in possible_heat_pump_cost_names
                if technology in cost_data.index
            ),
            None,
        )

        if heat_pump_cost_name is None:
            print("Warning: No ASHP cost found. Using zero heat pump costs.")
            heat_pump_capital_cost = 0.0
            heat_pump_marginal_cost = 0.0
        else:
            heat_pump_capital_cost = cost_data.at[heat_pump_cost_name, "fixed"]
            heat_pump_marginal_cost = cost_data.at[heat_pump_cost_name, "VOM"]

        n.add(
            "Link",
            f"{country_code}_ASHP",
            bus0=country_code,
            bus1=heat_bus,
            carrier="heat pump",
            p_nom_extendable=True,
            efficiency=heat_pump_cop,
            capital_cost=heat_pump_capital_cost,
            marginal_cost=heat_pump_marginal_cost,
        )

        # -----------------------------
        # Optional heat storage
        # -----------------------------
        if scenario.get("with_heat_storage", False):
            possible_heat_storage_cost_names = [
                "central water tank storage",
                "water tank storage",
                "hot water storage",
            ]

            heat_storage_cost_name = next(
                (
                    technology
                    for technology in possible_heat_storage_cost_names
                    if technology in cost_data.index
                ),
                None,
            )

            if heat_storage_cost_name is None:
                print(
                    "Warning: No heat storage cost found. "
                    "Using zero heat storage costs."
                )
                heat_storage_capital_cost = 0.0
            else:
                heat_storage_capital_cost = cost_data.at[
                    heat_storage_cost_name,
                    "fixed",
                ]

            # Heat storage bus
            n.add(
                "Bus",
                heat_storage_bus,
                carrier="heat storage",
            )

            # Heat energy store
            n.add(
                "Store",
                f"{country_code}_heat_store",
                bus=heat_storage_bus,
                carrier="heat storage",
                e_nom_extendable=True,
                e_cyclic=True,
                capital_cost=heat_storage_capital_cost,
            )

            # Charge link: heat bus -> heat storage bus
            n.add(
                "Link",
                f"{country_code}_heat_store_charger",
                bus0=heat_bus,
                bus1=heat_storage_bus,
                carrier="heat storage charger",
                p_nom_extendable=True,
                efficiency=1.0,
            )

            # Discharge link: heat storage bus -> heat bus
            n.add(
                "Link",
                f"{country_code}_heat_store_discharger",
                bus0=heat_storage_bus,
                bus1=heat_bus,
                carrier="heat storage discharger",
                p_nom_extendable=True,
                efficiency=1.0,
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

def custom_constraints(n: pypsa.Network, snapshots) -> None:
    """
    Add custom optimization constraints.

    Currently couples battery charger and discharger capacities.
    """
    if n.links.empty or not n.links.p_nom_extendable.any():
        return

    charger_links = n.links.index[n.links.index.str.contains("battery_charger")]
    discharger_links = n.links.index[n.links.index.str.contains("battery_discharger")]

    if len(charger_links) == 0 or len(discharger_links) == 0:
        return

    charger_p_nom = n.model["Link-p_nom"].loc[charger_links]
    discharger_p_nom = n.model["Link-p_nom"].loc[discharger_links]

    discharger_efficiency = n.links.loc[discharger_links, "efficiency"].values

    lhs = charger_p_nom - discharger_p_nom * discharger_efficiency

    n.model.add_constraints(lhs == 0, name="Link-battery_charger_ratio")


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

    if scenario["with_gas_network"]:
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

    if scenario["with_heat_sector"]:
        if heat_timeseries is None:
            raise ValueError("Heat sector is enabled, but no heat_timeseries were provided.")

        add_heat(
            n=n,
            cost_data=cost_data,
            scenario=scenario,
            heat_timeseries=heat_timeseries,
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