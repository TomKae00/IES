import os
from pathlib import Path
import pandas as pd
import pypsa

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

    def annuity_factor(row):
        return (
            calculate_annuity(row["lifetime"], financial_parameters["r"])
            + row["FOM"] / 100
        )

    cost_data["annuity_factor"] = cost_data.apply(annuity_factor, axis=1)
    cost_data["fixed"] = (
        cost_data["annuity_factor"] * cost_data["investment"] * number_of_years
    )

    return cost_data


def load_dk_timeseries(timeseries_file: str, year: str = "2016") -> dict:
    """
    Load Danish electricity demand and renewable generation data and
    compute capacity factors for solar, onshore wind and offshore wind.

    Parameters
    ----------
    timeseries_file : str
        Path to the CSV file containing hourly Danish electricity data.
    year : str, optional
        Year to extract from the dataset.

    Returns
    -------
    dict
        Dictionary containing electricity demand and renewable capacity factors.
    """
    raw_timeseries = pd.read_csv(timeseries_file)

    raw_timeseries["utc_timestamp"] = pd.to_datetime(
        raw_timeseries["utc_timestamp"], utc=True
    )
    raw_timeseries = raw_timeseries.set_index("utc_timestamp")
    raw_timeseries.index = raw_timeseries.index.tz_localize(None)

    yearly_timeseries = raw_timeseries.loc[f"{year}-01-01":f"{year}-12-31"]

    electricity_load = yearly_timeseries["DK_load_actual_entsoe_transparency"].copy()

    solar_capacity_factor = (
        yearly_timeseries["DK_solar_generation_actual"]
        / yearly_timeseries["DK_solar_capacity"]
    ).fillna(0).clip(0, 1)

    onshore_wind_capacity_factor = (
        yearly_timeseries["DK_wind_onshore_generation_actual"]
        / yearly_timeseries["DK_wind_onshore_capacity"]
    ).fillna(0).clip(0, 1)

    offshore_wind_capacity_factor = (
        yearly_timeseries["DK_wind_offshore_generation_actual"]
        / yearly_timeseries["DK_wind_offshore_capacity"]
    ).fillna(0).clip(0, 1)

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
    co2_price: float,
) -> float:
    """
    Calculate the marginal cost of a conventional generator in EUR/MWh_el.

    Parameters
    ----------
    cost_data : pd.DataFrame
        Prepared technology cost table.
    technology : str
        Generator technology, e.g. "OCGT", "CCGT", "coal", or "nuclear".
    co2_price : float
        CO2 price in EUR/tCO2.

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


def add_carriers(n: pypsa.Network) -> None:
    """
    Add carriers used in the system and define plotting colors.
    """

    carrier_data = {
        "AC": {"color": "#000000"},
        "solar": {"color": "#ffd92f"},
        "onwind": {"color": "#1b9e77"},
        "offwind": {"color": "#377eb8"},
        "gas": {"color": "#e41a1c"},
        "coal": {"color": "#4d4d4d"},
        "nuclear": {"color": "#984ea3"},
    }

    for carrier, attrs in carrier_data.items():
        if carrier not in n.carriers.index:
            n.add(
                "Carrier",
                carrier,
                **attrs,
            )


def attach_renewable_generators_dk(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
    timeseries_data: dict,
) -> None:
    """
    Attach renewable generators to the Danish one-node network.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    cost_data : pd.DataFrame
        Prepared technology cost table.
    timeseries_data : dict
        Dictionary containing demand and renewable capacity factor time series.
    """
    renewable_generators = {
        "solar": {
            "name": "DK_solar",
            "carrier": "solar",
            "capacity_factor": timeseries_data["solar_cf"],
        },
        "onwind": {
            "name": "DK_onshore_wind",
            "carrier": "onwind",
            "capacity_factor": timeseries_data["onshore_wind_cf"],
        },
        "offwind": {
            "name": "DK_offshore_wind",
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
            bus="DK_AC",
            carrier=generator_data["carrier"],
            p_nom_extendable=True,
            p_max_pu=generator_data["capacity_factor"],
            capital_cost=cost_data.at[technology, "fixed"],
            marginal_cost=cost_data.at[technology, "VOM"],
        )


def attach_conventional_generators_dk(
    n: pypsa.Network,
    cost_data: pd.DataFrame,
    co2_price: float,
) -> None:
    """
    Attach stylized conventional generators to the Danish one-node network.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network.
    cost_data : pd.DataFrame
        Prepared technology cost table.
    co2_price : float
        CO2 price in EUR/tCO2.
    """
    conventional_generators = {
        "CCGT": {"name": "DK_CCGT", "carrier": "CCGT"},
        "coal": {"name": "DK_coal", "carrier": "coal"},
        "nuclear": {"name": "DK_nuclear", "carrier": "nuclear"},
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
            bus="DK_AC",
            carrier=generator_data["carrier"],
            p_nom_extendable=True,
            capital_cost=cost_data.at[technology, "fixed"],
            marginal_cost=marginal_cost,
            efficiency=cost_data.at[technology, "efficiency"],
        )


def create_network_dk(
    cost_data: pd.DataFrame,
    timeseries_data: dict,
    co2_price: float,
) -> pypsa.Network:
    """
    Create a one-node PyPSA network for Denmark with electricity demand,
    renewable generators and conventional generators.

    Parameters
    ----------
    cost_data : pd.DataFrame
        Prepared technology cost table.
    timeseries_data : dict
        Dictionary containing electricity demand and renewable capacity factors.
    co2_price : float
        CO2 price in EUR/tCO2.

    Returns
    -------
    pypsa.Network
        Danish one-node electricity system model.
    """
    n = pypsa.Network()

    electricity_load = timeseries_data["load"]
    n.set_snapshots(electricity_load.index)

    add_carriers(n)

    n.add(
        "Bus",
        "DK_AC",
        carrier="AC",
    )

    n.add(
        "Load",
        "DK_electricity_demand",
        bus="DK_AC",
        carrier="AC",
        p_set=electricity_load,
    )

    attach_renewable_generators_dk(
        n=n,
        cost_data=cost_data,
        timeseries_data=timeseries_data,
    )

    attach_conventional_generators_dk(
        n=n,
        cost_data=cost_data,
        co2_price=co2_price,
    )

    return n


def optimize_and_save_network(
    n: pypsa.Network,
    output_file: str,
) -> None:
    """
    Optimize the PyPSA network with Gurobi and save it as a NetCDF file.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network to optimize.
    output_file : str
        Path to the output .nc file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n.optimize(solver_name="gurobi")
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

    cost_file = f"cost_data/costs_{financial_parameters['year']}.csv"
    timeseries_file = "Data/time_series_60min_singleindex_filtered_DK.csv"
    output_file = "results/dk_network_2016.nc"

    cost_data = prepare_costs(
        cost_file=cost_file,
        financial_parameters=financial_parameters,
        number_of_years=financial_parameters["nyears"],
    )

    timeseries_data = load_dk_timeseries(
        timeseries_file=timeseries_file,
        year="2016",
    )

    n = create_network_dk(
        cost_data=cost_data,
        timeseries_data=timeseries_data,
        co2_price=financial_parameters["co2_price"],
    )

    print("\nBUSES")
    print(n.buses.dtypes)

    print("\nGENERATORS")
    print(n.generators.dtypes)

    print("\nLOADS")
    print(n.loads.dtypes)

    print("\nINDEX DTYPES")
    print("buses index:", n.buses.index.dtype)
    print("generators index:", n.generators.index.dtype)
    print("loads index:", n.loads.index.dtype)

    optimize_and_save_network(
        n=n,
        output_file=output_file,
    )

    print("\nOptimized capacities [MW]:")
    print(n.generators.p_nom_opt)

    print("\nObjective value:")
    print(n.objective)