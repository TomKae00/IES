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

    Electricity demand is always taken from 2016 and mapped to the selected
    model year. Renewable generation and capacity data are taken from the
    selected weather year.

    Missing hourly values in the required input columns are interpolated.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary with:
        - load
        - solar_cf
        - onshore_wind_cf
        - offshore_wind_cf
    """
    demand_year = 2016
    weather_year = int(year)

    raw_timeseries = pd.read_csv(timeseries_file)

    raw_timeseries["utc_timestamp"] = pd.to_datetime(
        raw_timeseries["utc_timestamp"],
        utc=True,
    )

    raw_timeseries = raw_timeseries.set_index("utc_timestamp")
    raw_timeseries.index = raw_timeseries.index.tz_localize(None)

    load_column = f"{country_code}_load_actual_entsoe_transparency"

    renewable_columns = [
        f"{country_code}_solar_generation_actual",
        f"{country_code}_solar_capacity",
        f"{country_code}_wind_onshore_generation_actual",
        f"{country_code}_wind_onshore_capacity",
        f"{country_code}_wind_offshore_generation_actual",
        f"{country_code}_wind_offshore_capacity",
    ]

    if load_column not in raw_timeseries.columns:
        raise KeyError(
            f"Load column '{load_column}' not found in {timeseries_file}."
        )

    # -----------------------------
    # Load demand from fixed demand year
    # -----------------------------
    demand_data = raw_timeseries.loc[
        f"{demand_year}-01-01":f"{demand_year}-12-31",
        [load_column],
    ].copy()

    demand_data[load_column] = (
        demand_data[load_column]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    demand_data[load_column] = pd.to_numeric(
        demand_data[load_column],
        errors="coerce",
    )

    demand_index = pd.date_range(
        start=f"{demand_year}-01-01 00:00:00",
        end=f"{demand_year}-12-31 23:00:00",
        freq="h",
    )

    demand_data = demand_data.reindex(demand_index)

    missing_demand_timestamps = demand_data.index[
        demand_data[load_column].isna()
    ]

    if len(missing_demand_timestamps) > 0:
        print(
            f"Warning: electricity demand time series for {country_code} has "
            f"{len(missing_demand_timestamps)} missing timestamps in {demand_year}. "
            "Missing values are interpolated."
        )
        print(missing_demand_timestamps)

    demand_data = demand_data.interpolate(method="time")
    demand_data = demand_data.ffill().bfill()

    if demand_data[load_column].isna().any():
        raise ValueError(
            f"Load column '{load_column}' still contains NaN values after interpolation."
        )

    # -----------------------------
    # Load renewable data from selected weather year
    # -----------------------------
    weather_required_columns = [
        column for column in renewable_columns if column in raw_timeseries.columns
    ]

    weather_data = raw_timeseries.loc[
        f"{weather_year}-01-01":f"{weather_year}-12-31",
        weather_required_columns,
    ].copy()

    for column in weather_data.columns:
        weather_data[column] = (
            weather_data[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        weather_data[column] = pd.to_numeric(
            weather_data[column],
            errors="coerce",
        )

    target_index = pd.date_range(
        start=f"{weather_year}-01-01 00:00:00",
        end=f"{weather_year}-12-31 23:00:00",
        freq="h",
    )

    weather_data = weather_data.reindex(target_index)

    missing_weather_timestamps = weather_data.index[
        weather_data.isna().any(axis=1)
    ]

    if len(missing_weather_timestamps) > 0:
        print(
            f"Warning: renewable time series for {country_code} has "
            f"{len(missing_weather_timestamps)} timestamps with missing values "
            f"in {weather_year}. Missing values are interpolated."
        )
        print(missing_weather_timestamps)

    weather_data = weather_data.interpolate(method="time")
    weather_data = weather_data.ffill().bfill()

    # -----------------------------
    # Map fixed 2016 demand onto selected model year
    # -----------------------------
    if len(target_index) <= len(demand_data):
        mapped_demand = demand_data.iloc[: len(target_index)].copy()
    else:
        missing_hours = len(target_index) - len(demand_data)

        extension = demand_data.iloc[-24:].copy()
        repeats = int(-(-missing_hours // 24))
        extension = pd.concat([extension] * repeats).iloc[:missing_hours]

        mapped_demand = pd.concat([demand_data, extension])

    mapped_demand = mapped_demand.iloc[: len(target_index)].copy()
    mapped_demand.index = target_index

    electricity_load = mapped_demand[load_column].copy()

    zero_series = pd.Series(0.0, index=target_index)

    def capacity_factor_from_columns(
        generation_column: str,
        capacity_column: str,
    ) -> pd.Series:
        """
        Calculate capacity factor as generation / installed capacity.

        If the required columns are missing, return a zero series.
        """
        if (
            generation_column not in weather_data.columns
            or capacity_column not in weather_data.columns
        ):
            return zero_series.copy()

        capacity = weather_data[capacity_column].replace(0, pd.NA)

        capacity_factor = (
            weather_data[generation_column] / capacity
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

    Heat demand uses:
        {country_code}_heat_demand_total

    Heat pump COP uses:
        {country_code}_COP_ASHP_floor

    The heat input file only contains reliable data for 2015.
    Therefore, the 2015 heat demand and COP profiles are reused for all model years
    and mapped onto the requested model year.

    Norway is not included in the heat data file. Therefore, Danish heat demand
    and COP profiles are used as a proxy for Norway.
    """
    heat_data_year = 2015
    target_year = int(year)

    raw_heat_data = pd.read_csv(
        heat_file,
        sep=None,
        engine="python",
    )

    raw_heat_data["utc_timestamp"] = pd.to_datetime(
        raw_heat_data["utc_timestamp"],
        utc=True,
    )

    raw_heat_data = raw_heat_data.set_index("utc_timestamp")
    raw_heat_data.index = raw_heat_data.index.tz_localize(None)

    required_columns = []

    for country_code in countries:
        data_country_code = "DK" if country_code == "NO" else country_code

        required_columns.extend(
            [
                f"{data_country_code}_heat_demand_total",
                f"{data_country_code}_COP_ASHP_floor",
            ]
        )

    required_columns = list(dict.fromkeys(required_columns))

    missing_columns = [
        column for column in required_columns
        if column not in raw_heat_data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"The following required heat columns are missing in {heat_file}: "
            f"{missing_columns}"
        )

    source_heat_data = raw_heat_data.loc[
        f"{heat_data_year}-01-01":f"{heat_data_year}-12-31",
        required_columns,
    ].copy()

    for column in source_heat_data.columns:
        source_heat_data[column] = (
            source_heat_data[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        source_heat_data[column] = pd.to_numeric(
            source_heat_data[column],
            errors="coerce",
        )

    source_index = pd.date_range(
        start=f"{heat_data_year}-01-01 00:00:00",
        end=f"{heat_data_year}-12-31 23:00:00",
        freq="h",
    )

    source_heat_data = source_heat_data.reindex(source_index)

    missing_timestamps = source_heat_data.index[
        source_heat_data.isna().any(axis=1)
    ]

    if len(missing_timestamps) > 0:
        print(
            f"Warning: heat source time series has {len(missing_timestamps)} "
            f"missing timestamps in {heat_data_year}. "
            "Missing values are interpolated."
        )
        print(missing_timestamps)

    source_heat_data = source_heat_data.interpolate(method="time")
    source_heat_data = source_heat_data.ffill().bfill()

    if source_heat_data.isna().any().any():
        missing_columns_after_interpolation = source_heat_data.columns[
            source_heat_data.isna().any()
        ].tolist()

        raise ValueError(
            "Heat source time series still contains NaN values after interpolation. "
            f"Affected columns: {missing_columns_after_interpolation}"
        )

    target_index = pd.date_range(
        start=f"{target_year}-01-01 00:00:00",
        end=f"{target_year}-12-31 23:00:00",
        freq="h",
    )

    if len(target_index) <= len(source_heat_data):
        mapped_heat_data = source_heat_data.iloc[: len(target_index)].copy()
    else:
        missing_hours = len(target_index) - len(source_heat_data)

        extension = source_heat_data.iloc[-24:].copy()
        repeats = int(-(-missing_hours // 24))
        extension = pd.concat([extension] * repeats).iloc[:missing_hours]

        mapped_heat_data = pd.concat([source_heat_data, extension])

    mapped_heat_data = mapped_heat_data.iloc[: len(target_index)].copy()
    mapped_heat_data.index = target_index

    heat_timeseries = {}

    for country_code in countries:
        data_country_code = "DK" if country_code == "NO" else country_code

        heat_demand_column = f"{data_country_code}_heat_demand_total"
        cop_column = f"{data_country_code}_COP_ASHP_floor"

        heat_timeseries[country_code] = {
            "heat_demand": mapped_heat_data[heat_demand_column].copy(),
            "ashp_cop": mapped_heat_data[cop_column].copy(),
        }

    return heat_timeseries