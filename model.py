import os
from pathlib import Path

import pandas as pd
import pypsa

import logging

gurobi_logger = logging.getLogger("gurobipy")
gurobi_logger.setLevel(logging.CRITICAL)
gurobi_logger.disabled = True
gurobi_logger.propagate = False
gurobi_logger.handlers.clear()


print(os.getcwd())


def calculate_annuity(lifetime, discount_rate):
    """
    Calculate the annuity factor for an asset with lifetime n years and
    discount rate r, e.g. annuity(20, 0.05) * 20 = 1.6.

    Parameters
    ----------
    lifetime : float or pd.Series
        Asset lifetime in years.
    discount_rate : float or pd.Series
        Discount rate.

    Returns
    -------
    float or pd.Series
        Annuity factor.
    """
    if isinstance(discount_rate, pd.Series):
        return pd.Series(1 / lifetime, index=discount_rate.index).where(
            discount_rate == 0,
            discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime),
        )
    elif discount_rate > 0:
        return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)
    else:
        return 1 / lifetime


def prepare_costs(cost_file, financial_parameters, number_of_years):
    """
    Load and prepare technology cost data.

    Parameters
    ----------
    cost_file : str
        Path to the technology cost CSV file.
    financial_parameters : dict
        Dictionary containing financial assumptions.
    number_of_years : int or float
        Number of years represented in the optimisation.

    Returns
    -------
    pd.DataFrame
        Prepared cost table in wide format.
    """
    cost_data = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    cost_data.loc[cost_data.unit.str.contains("/kW", na=False), "value"] *= 1e3

    cost_data = (
        cost_data.loc[:, "value"]
        .unstack(level=1)
        .groupby("technology")
        .sum(min_count=1)
    )

    cost_data = cost_data.fillna(financial_parameters["fill_values"])

    valid_investment_rows = (
        cost_data["investment"].fillna(0) > 0
    ) & (
        cost_data["lifetime"].fillna(0) > 0
    )

    cost_data["annuity_factor"] = 0.0
    cost_data.loc[valid_investment_rows, "annuity_factor"] = (
        cost_data.loc[valid_investment_rows].apply(
            lambda row: calculate_annuity(row["lifetime"], financial_parameters["r"])
            + row["FOM"] / 100,
            axis=1,
        )
    )

    cost_data["fixed"] = 0.0
    cost_data.loc[valid_investment_rows, "fixed"] = (
        cost_data.loc[valid_investment_rows, "annuity_factor"]
        * cost_data.loc[valid_investment_rows, "investment"]
        * number_of_years
    )

    return cost_data


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
        if generation_column in yearly_timeseries.columns and capacity_column in yearly_timeseries.columns:
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

def calculate_conventional_marginal_cost(
    cost_data: pd.DataFrame,
    technology: str,
    co2_price: float = 0.0,
) -> float:
    """
    Calculate the marginal cost of a conventional generator in EUR/MWh_el.
    
    Includes CO2 cost if co2_price > 0.

    Parameters
    ----------
    cost_data : pd.DataFrame
        Prepared technology cost table.
    technology : str
        Generator technology, e.g. "CCGT", "coal", or "nuclear".
    co2_price : float, optional
        CO2 price in EUR/tCO2. Default is 0.0 (no CO2 cost).

    Returns
    -------
    float
        Marginal generation cost in EUR/MWh_el.
    """
    fuel_lookup = {
        "CCGT": "gas",
        "coal": "coal",
        "nuclear": "uranium",
    }

    fuel_technology = fuel_lookup[technology]

    efficiency = cost_data.at[technology, "efficiency"]
    variable_om_cost = cost_data.at[technology, "VOM"]
    fuel_cost = cost_data.at[fuel_technology, "fuel"]

    if "CO2 intensity" in cost_data.columns and fuel_technology in cost_data.index:
        co2_intensity = cost_data.at[fuel_technology, "CO2 intensity"]
        co2_cost = co2_intensity / efficiency * co2_price
    else:
        co2_cost = 0.0

    marginal_cost = fuel_cost / efficiency + variable_om_cost + co2_cost

    return marginal_cost


def attach_country_bus_and_load(
    n: pypsa.Network,
    country_code: str,
    timeseries_data: dict,
) -> None:
    """
    Attach one country bus and load to the network.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    country_code : str
        Country code, e.g. "DK", "DE", "SE", "NO".
    timeseries_data : dict
        Dictionary containing the load time series.
    """
    bus_name = f"{country_code}"
    load_name = f"{country_code}_electricity_demand"

    n.add(
        "Bus",
        bus_name,
        carrier="electricity",
    )

    n.add(
        "Load",
        load_name,
        bus=bus_name,
        carrier="electricity",
        p_set=timeseries_data["load"],
    )


def add_carriers(n: pypsa.Network, cost_data: pd.DataFrame = None) -> None:
    """
    Add carriers used in the system and define plotting colors.
    
    For gas and coal carriers, also set co2_emissions attribute (tCO2/MWh_th).

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    cost_data : pd.DataFrame, optional
        Technology cost data containing CO2 intensity values.
    """
    # Get CO2 intensities from cost data if available
    gas_co2 = 0.0
    coal_co2 = 0.0
    
    if cost_data is not None and "CO2 intensity" in cost_data.columns:
        if "gas" in cost_data.index:
            gas_co2 = cost_data.at["gas", "CO2 intensity"]
        if "coal" in cost_data.index:
            coal_co2 = cost_data.at["coal", "CO2 intensity"]
    
    carrier_data = {
        "electricity": {"color": "#4C566A", "co2_emissions": 0.0},
        "solar": {"color": "#EBCB3B", "co2_emissions": 0.0},
        "onwind": {"color": "#5AA469", "co2_emissions": 0.0},
        "offwind": {"color": "#2E86AB", "co2_emissions": 0.0},
        "gas": {"color": "#D08770", "co2_emissions": gas_co2},
        "coal": {"color": "#5C5C5C", "co2_emissions": coal_co2},
        "nuclear": {"color": "#8F6BB3", "co2_emissions": 0.0},
        "battery": {"color": "#E67E22", "co2_emissions": 0.0},
        "battery charger": {"color": "#C06C84", "co2_emissions": 0.0},
        "battery discharger": {"color": "#6C5B7B", "co2_emissions": 0.0},
    }

    for carrier, attrs in carrier_data.items():
        if carrier not in n.carriers.index:
            n.add(
                "Carrier",
                carrier,
                **attrs,
            )
        else:
            # Update co2_emissions for existing carriers
            if "co2_emissions" in attrs:
                n.carriers.at[carrier, "co2_emissions"] = attrs["co2_emissions"]


def attach_renewable_generators(
    n: pypsa.Network,
    country_code: str,
    cost_data: pd.DataFrame,
    timeseries_data: dict,
) -> None:
    """
    Attach renewable generators to one country node.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    country_code : str
        Country code, e.g. "DK", "DE", "SE", "NO".
    cost_data : pd.DataFrame
        Prepared technology cost table.
    timeseries_data : dict
        Dictionary containing renewable capacity factor time series.
    """
    bus_name = f"{country_code}"

    renewable_generators = {
        "solar": {
            "name": f"{country_code}_solar",
            "carrier": "solar",
            "capacity_factor": timeseries_data["solar_cf"],
        },
        "onwind": {
            "name": f"{country_code}_onshore_wind",
            "carrier": "onwind",
            "capacity_factor": timeseries_data["onshore_wind_cf"],
        },
        "offwind": {
            "name": f"{country_code}_offshore_wind",
            "carrier": "offwind",
            "capacity_factor": timeseries_data["offshore_wind_cf"],
        },
    }

    for technology, generator_data in renewable_generators.items():
        if technology not in cost_data.index:
            continue

        n.add(
            "Generator",
            generator_data["name"],
            bus=bus_name,
            carrier=generator_data["carrier"],
            p_nom_extendable=True,
            p_max_pu=generator_data["capacity_factor"],
            capital_cost=cost_data.at[technology, "fixed"],
            marginal_cost=cost_data.at[technology, "VOM"],
        )


def attach_conventional_generators(
    n: pypsa.Network,
    country_code: str,
    cost_data: pd.DataFrame,
    co2_price: float = 0.0,
) -> None:
    """
    Attach stylized conventional generators to one country node.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    country_code : str
        Country code, e.g. "DK", "DE", "SE", "NO".
    cost_data : pd.DataFrame
        Prepared technology cost table.
    co2_price : float, optional
        CO2 price in EUR/tCO2. Default is 0.0 (no CO2 cost).
    """
    bus_name = f"{country_code}"

    conventional_generators = {
        "CCGT": {"name": f"{country_code}_CCGT", "carrier": "gas"},
        "coal": {"name": f"{country_code}_coal", "carrier": "coal"},
        "nuclear": {"name": f"{country_code}_nuclear", "carrier": "nuclear"},
    }

    for technology, generator_data in conventional_generators.items():
        if technology not in cost_data.index:
            continue

        marginal_cost = calculate_conventional_marginal_cost(
            cost_data=cost_data,
            technology=technology,
            co2_price=co2_price,
        )

        n.add(
            "Generator",
            generator_data["name"],
            bus=bus_name,
            carrier=generator_data["carrier"],
            p_nom_extendable=True,
            capital_cost=cost_data.at[technology, "fixed"],
            marginal_cost=marginal_cost,
            efficiency=cost_data.at[technology, "efficiency"],
        )


def attach_battery_storage_dk(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
) -> None:
    """
    Attach a battery storage system to the Danish one-node network.

    The battery is represented by:
    - one battery bus
    - one Store for the energy capacity
    - one charging Link
    - one discharging Link

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    cost_data : pd.DataFrame
        Prepared technology cost table.
    """
    battery_bus = "DK_battery"

    n.add(
        "Bus",
        battery_bus,
        carrier="battery",
    )

    n.add(
        "Store",
        "DK_battery_store",
        bus=battery_bus,
        carrier="battery",
        e_cyclic=True,
        e_nom_extendable=True,
        capital_cost=cost_data.at["battery storage", "fixed"],
    )

    n.add(
        "Link",
        "DK_battery_charger",
        bus0="DK",
        bus1=battery_bus,
        carrier="battery charger",
        efficiency=cost_data.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=cost_data.at["battery inverter", "fixed"],
        marginal_cost=cost_data.at["battery inverter", "VOM"],
        p_nom_extendable=True,
    )

    n.add(
        "Link",
        "DK_battery_discharger",
        bus0=battery_bus,
        bus1="DK",
        carrier="battery discharger",
        efficiency=cost_data.at["battery inverter", "efficiency"] ** 0.5,
        marginal_cost=cost_data.at["battery inverter", "VOM"],
        p_nom_extendable=True,
    )


def attach_interconnectors_dk_region(n: pypsa.Network) -> None:
    """
    Attach fixed cross-border interconnectors for task d).

    The real assets are mostly HVDC, but they are approximated here as
    400 kV HVAC lines with x = 0.1 according to the assignment.
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


def create_regional_network(
    cost_data: pd.DataFrame,
    all_timeseries_data: dict,
    with_battery_storage: bool,
    with_interconnectors: bool,
    co2_price: float = 0.0,
) -> pypsa.Network:
    """
    Create a regional PyPSA network with Denmark and neighbouring countries.

    Parameters
    ----------
    cost_data : pd.DataFrame
        Prepared technology cost table.
    all_timeseries_data : dict
        Dictionary mapping country codes to their time series data.
    with_battery_storage : bool
        Whether to include battery storage for Denmark.
    with_interconnectors : bool
        Whether to include interconnectors.
    co2_price : float, optional
        CO2 price in EUR/tCO2. Default is 0.0 (no CO2 cost).

    Returns
    -------
    pypsa.Network
        Regional electricity system model.
    """
    n = pypsa.Network()

    reference_country = list(all_timeseries_data.keys())[0]
    n.set_snapshots(all_timeseries_data[reference_country]["load"].index)

    add_carriers(n, cost_data=cost_data)

    for country_code, timeseries_data in all_timeseries_data.items():
        attach_country_bus_and_load(
            n=n,
            country_code=country_code,
            timeseries_data=timeseries_data,
        )

        attach_renewable_generators(
            n=n,
            country_code=country_code,
            cost_data=cost_data,
            timeseries_data=timeseries_data,
        )

        attach_conventional_generators(
            n=n,
            country_code=country_code,
            cost_data=cost_data,
            co2_price=co2_price,
        )

    if with_battery_storage:
        attach_battery_storage_dk(
            n=n,
            cost_data=cost_data,
        )

    if with_interconnectors:
        attach_interconnectors_dk_region(n)

    return n


def apply_battery_ratio_constraint(n: pypsa.Network, sns) -> None:
    """
    Add constraint: charger and discharger power capacities must be equal.
    """
    if n.links.empty or not n.links.p_nom_extendable.any():
        return

    charger_links = n.links.index[n.links.index.str.contains("battery_charger")]
    discharger_links = n.links.index[n.links.index.str.contains("battery_discharger")]

    if len(charger_links) == 0 or len(discharger_links) == 0:
        return

    charger_p_nom = n.model["Link-p_nom"].loc[charger_links]
    discharger_p_nom = n.model["Link-p_nom"].loc[discharger_links]

    lhs = charger_p_nom - discharger_p_nom
    n.model.add_constraints(lhs == 0, name="Link-battery_charger_ratio")

def optimize_and_save_network(
    n: pypsa.Network,
    output_file: str,
    co2_limit_mt: float = None,
) -> None:
    """
    Optimize the PyPSA network with Gurobi and save it as a NetCDF file.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network to optimize.
    output_file : str
        Path to the output .nc file.
    co2_limit_mt : float, optional
        CO2 emission limit in Mt CO2 per year. If None, no CO2 constraint is applied.
        Uses PyPSA's 
Constraint feature.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add CO2 global constraint if specified
    if co2_limit_mt is not None:
        n.add(
            "GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit_mt * 1e6,  # Convert Mt to tCO2
        )

    solver_options = {}

    n.optimize(
        n.snapshots,
        extra_functionality=apply_battery_ratio_constraint,
        solver_name="gurobi",
        solver_options=solver_options,
    )

    n.export_to_netcdf(output_path)

    print(f"Optimized network saved to: {output_path}")


if __name__ == "__main__":
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
     "countries": ["DK"] #, "DE", "SE", "NO"
     }

    file_paths = {
        "cost_file": f"cost_data/costs_{financial_parameters['year']}.csv",
        "timeseries_file": "Data/time_series_60min_singleindex_alldata.csv",
    }

    cost_data = prepare_costs(
        cost_file=file_paths["cost_file"],
        financial_parameters=financial_parameters,
        number_of_years=financial_parameters["nyears"],
    )

    all_timeseries_data = {}

    for country_code in scenario_parameters["countries"]:
        all_timeseries_data[country_code] = load_country_timeseries(
            timeseries_file=file_paths["timeseries_file"],
            country_code=country_code,
            year=scenario_parameters["weather_year"],
        )

    n = create_regional_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        with_battery_storage=scenario_parameters["with_battery_storage"],
        with_interconnectors=scenario_parameters["with_interconnectors"],
        co2_price=financial_parameters["co2_price"],
    )

    # Optimize and save network
    optimize_and_save_network(
        n=n,
        output_file="results/optimized_network.nc",
    )

    print("\n" + "=" * 80)
    print("BASIC NETWORK INFO")
    print("=" * 80)
    print(f"Buses: {len(n.buses)}")
    print(f"Generators: {len(n.generators)}")
    print(f"Loads: {len(n.loads)}")
    print(f"Snapshots: {len(n.snapshots)}")

    print("\nCarrier CO2 emissions:")
    print(n.carriers[["co2_emissions"]])

    
    
    print(cost_data.loc[["gas", "coal"], ["fuel", "efficiency", "VOM", "CO2 intensity"]])
    print("Coal marginal cost:",
          calculate_conventional_marginal_cost(cost_data, "coal", financial_parameters["co2_price"]))
    print("CCGT marginal cost:",
          calculate_conventional_marginal_cost(cost_data, "CCGT", financial_parameters["co2_price"]))
    print("Nuclear marginal cost:",
          calculate_conventional_marginal_cost(cost_data, "nuclear", financial_parameters["co2_price"]))    
    