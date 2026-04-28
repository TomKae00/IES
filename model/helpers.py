from pathlib import Path
import logging

import pandas as pd


def silence_gurobi_logger() -> None:
    gurobi_logger = logging.getLogger("gurobipy")
    gurobi_logger.setLevel(logging.CRITICAL)
    gurobi_logger.disabled = True
    gurobi_logger.propagate = False
    gurobi_logger.handlers.clear()


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def calculate_annuity(lifetime, discount_rate):
    if isinstance(discount_rate, pd.Series):
        return pd.Series(1 / lifetime, index=discount_rate.index).where(
            discount_rate == 0,
            discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime),
        )

    if discount_rate > 0:
        return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)

    return 1 / lifetime


def prepare_costs(
    cost_file: str,
    financial_parameters: dict,
    number_of_years: int | float,
) -> pd.DataFrame:
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
        (cost_data["investment"].fillna(0) > 0)
        & (cost_data["lifetime"].fillna(0) > 0)
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
    year: str | int,
) -> dict[str, pd.Series]:
    """
    Load electricity demand and renewable capacity factors for one country.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary with:
        - load
        - solar_cf
        - onshore_wind_cf
        - offshore_wind_cf
    """
    raw_timeseries = pd.read_csv(timeseries_file)

    raw_timeseries["utc_timestamp"] = pd.to_datetime(
        raw_timeseries["utc_timestamp"],
        utc=True,
    )

    raw_timeseries = raw_timeseries.set_index("utc_timestamp")
    raw_timeseries.index = raw_timeseries.index.tz_localize(None)

    yearly_timeseries = raw_timeseries.loc[f"{year}-01-01":f"{year}-12-31"].copy()

    load_column = f"{country_code}_load_actual_entsoe_transparency"

    if load_column not in yearly_timeseries.columns:
        raise KeyError(
            f"Load column '{load_column}' not found in {timeseries_file}."
        )

    electricity_load = yearly_timeseries[load_column].copy()

    zero_series = pd.Series(0.0, index=yearly_timeseries.index)

    def capacity_factor_from_columns(
        generation_column: str,
        capacity_column: str,
    ) -> pd.Series:
        """
        Calculate capacity factor as generation / installed capacity.

        If the required columns are missing, return a zero series.
        """
        if (
            generation_column not in yearly_timeseries.columns
            or capacity_column not in yearly_timeseries.columns
        ):
            return zero_series.copy()

        capacity = yearly_timeseries[capacity_column].replace(0, pd.NA)

        capacity_factor = (
            yearly_timeseries[generation_column] / capacity
        ).fillna(0.0)

        return capacity_factor.clip(lower=0.0, upper=1.0)

    return {
        "load": electricity_load,
        "solar_cf": capacity_factor_from_columns(
            generation_column=f"{country_code}_solar_generation_actual",
            capacity_column=f"{country_code}_solar_capacity",
        ),
        "onshore_wind_cf": capacity_factor_from_columns(
            generation_column=f"{country_code}_wind_onshore_generation_actual",
            capacity_column=f"{country_code}_wind_onshore_capacity",
        ),
        "offshore_wind_cf": capacity_factor_from_columns(
            generation_column=f"{country_code}_wind_offshore_generation_actual",
            capacity_column=f"{country_code}_wind_offshore_capacity",
        ),
    }


def load_all_countries_timeseries(
    timeseries_file: str,
    countries: list[str],
    year: str | int,
) -> dict[str, dict[str, pd.Series]]:
    """
    Load electricity demand and renewable capacity factors for all modeled countries.

    Returns
    -------
    dict[str, dict[str, pd.Series]]
        Nested dictionary:

        {
            "DK": {
                "load": ...,
                "solar_cf": ...,
                "onshore_wind_cf": ...,
                "offshore_wind_cf": ...,
            },
            "DE": {
                ...
            },
        }
    """
    all_timeseries_data = {}

    for country_code in countries:
        all_timeseries_data[country_code] = load_country_timeseries(
            timeseries_file=timeseries_file,
            country_code=country_code,
            year=year,
        )

    return all_timeseries_data


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
        Generator technology, e.g. "CCGT", "coal", or "nuclear".
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


def load_heat_timeseries(
    heat_file: str,
    countries: list[str],
    year: str | int,
) -> dict[str, dict[str, pd.Series]]:
    """
    Load heat demand and air-source heat pump COP time series for all modeled countries.

    The heat demand uses:
        {country_code}_heat_demand_total

    The heat pump COP uses:
        {country_code}_COP_ASHP_floor

    Norway is not included in the heat data file. Therefore, Danish heat demand
    and COP profiles are used as a proxy for Norway.
    """
    raw_heat_data = pd.read_csv(heat_file)

    raw_heat_data["utc_timestamp"] = pd.to_datetime(
        raw_heat_data["utc_timestamp"],
        utc=True,
    )

    raw_heat_data = raw_heat_data.set_index("utc_timestamp")
    raw_heat_data.index = raw_heat_data.index.tz_localize(None)

    yearly_heat_data = raw_heat_data.loc[f"{year}-01-01":f"{year}-12-31"].copy()

    heat_timeseries = {}

    for country_code in countries:
        # Norway is approximated with Danish data because NO is missing.
        data_country_code = "DK" if country_code == "NO" else country_code

        heat_demand_column = f"{data_country_code}_heat_demand_total"
        cop_column = f"{data_country_code}_COP_ASHP_floor"

        if heat_demand_column not in yearly_heat_data.columns:
            raise KeyError(
                f"Heat demand column '{heat_demand_column}' not found in {heat_file}."
            )

        if cop_column not in yearly_heat_data.columns:
            raise KeyError(
                f"COP column '{cop_column}' not found in {heat_file}."
            )

        heat_timeseries[country_code] = {
            "heat_demand": yearly_heat_data[heat_demand_column].copy(),
            "ashp_cop": yearly_heat_data[cop_column].copy(),
        }

    return heat_timeseries
