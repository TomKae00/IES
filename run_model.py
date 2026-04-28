from pathlib import Path

from model.helpers import (
    silence_gurobi_logger,
    prepare_costs,
    load_all_countries_timeseries,
    load_heat_timeseries,
)

from model.model import (
    create_network,
    optimize_and_save_network,
    print_model_summary,
)

from model.scenarios import (
    FINANCIAL_PARAMETERS,
    FILE_PATHS,
    SCENARIOS,
)


# Change this to run another scenario
ACTIVE_SCENARIO = "sector_coupling"


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
    )

    print_model_summary(n)


if __name__ == "__main__":
    main()