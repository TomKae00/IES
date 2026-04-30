from pathlib import Path
import sys

import pandas as pd
import pypsa


# =========================================================
# PATH SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.helpers import prepare_costs  # noqa: E402
from model.scenarios import FINANCIAL_PARAMETERS, FILE_PATHS  # noqa: E402


# =========================================================
# CONFIG
# =========================================================

NETWORK_FILE = PROJECT_ROOT / "results" / "networks" / "sector_coupling_heat_2016.nc"
COST_FILE = PROJECT_ROOT / FILE_PATHS["cost_file"]

OUTPUT_DIR = PROJECT_ROOT / "results" / "component_inputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TECHNOLOGIES = {
    "Solar": "solar",
    "Onshore wind": "onwind",
    "Offshore wind": "offwind",
    "CCGT": "CCGT",
    "Coal": "coal",
    "Nuclear": "nuclear",
    "Battery inverter": "battery inverter",
    "Battery storage": "battery storage",
    "Electrolysis": "electrolysis",
    "H2 turbine / OCGT": "OCGT",
    "H2 fuel cell": "fuel cell",
    "H2 storage": "hydrogen storage tank type 1 including compressor",
    "Heat pump": "decentral air-sourced heat pump",
    "Resistive heater": "decentral resistive heater",
    "Gas boiler": "decentral gas boiler",
    "Water tank storage": "decentral water tank storage",
    "Water tank charger": "decentral water tank charger",
    "Water tank discharger": "decentral water tank discharger",
}


# =========================================================
# GENERAL HELPERS
# =========================================================

def safe_select(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Select only columns that exist.
    """
    existing_columns = [column for column in columns if column in df.columns]
    return df[existing_columns].copy()


def print_section(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def save_and_print(df: pd.DataFrame, name: str) -> None:
    """
    Print and save dataframe.
    """
    print_section(name)

    if df.empty:
        print("No entries found.")
        return

    print(df.to_string())

    output_path = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}.csv"
    df.to_csv(output_path)
    print(f"\nSaved to: {output_path}")


def get_prepared_value(
    prepared_costs: pd.DataFrame,
    technology: str,
    parameter: str,
):
    if technology not in prepared_costs.index:
        return None

    if parameter not in prepared_costs.columns:
        return None

    value = prepared_costs.at[technology, parameter]

    if pd.isna(value):
        return None

    return value


def build_technology_assumption_table(prepared_costs: pd.DataFrame) -> pd.DataFrame:
    """
    Build the techno-economic input table from prepared costs.
    """
    rows = []

    for label, technology in TECHNOLOGIES.items():
        investment = get_prepared_value(prepared_costs, technology, "investment")
        lifetime = get_prepared_value(prepared_costs, technology, "lifetime")
        fom = get_prepared_value(prepared_costs, technology, "FOM")
        vom = get_prepared_value(prepared_costs, technology, "VOM")
        efficiency = get_prepared_value(prepared_costs, technology, "efficiency")
        fuel = get_prepared_value(prepared_costs, technology, "fuel")
        co2_intensity = get_prepared_value(prepared_costs, technology, "CO2 intensity")
        fixed = get_prepared_value(prepared_costs, technology, "fixed")

        investment_eur_per_mw = None
        if investment is not None:
            investment_eur_per_mw = investment * 1000.0

        rows.append(
            {
                "Technology": label,
                "cost_key": technology,
                "Investment [EUR/MW]": investment_eur_per_mw,
                "Lifetime [yr]": lifetime,
                "FOM [%/yr]": fom,
                "VOM [EUR/MWh]": vom,
                "Efficiency [-]": efficiency,
                "Fuel price [EUR/MWh_th]": fuel,
                "CO2 intensity [tCO2/MWh_th]": co2_intensity,
                "Fixed [EUR/MW/yr]": fixed,
            }
        )

    return pd.DataFrame(rows)


def print_latex_rows(table: pd.DataFrame) -> None:
    """
    Print LaTeX rows for direct copy.
    """
    print_section("LATEX ROWS FOR TECHNO-ECONOMIC TABLE")

    def fmt(value, decimals):
        if value is None or pd.isna(value):
            return "--"
        return f"{value:.{decimals}f}"

    for _, row in table.iterrows():
        technology = row["Technology"]

        investment = fmt(row["Investment [EUR/MW]"], 1)
        lifetime = fmt(row["Lifetime [yr]"], 1)
        fom = fmt(row["FOM [%/yr]"], 4)
        vom = fmt(row["VOM [EUR/MWh]"], 4)
        efficiency = fmt(row["Efficiency [-]"], 3)
        fuel = fmt(row["Fuel price [EUR/MWh_th]"], 4)
        co2 = fmt(row["CO2 intensity [tCO2/MWh_th]"], 4)
        fixed = fmt(row["Fixed [EUR/MW/yr]"], 1)

        print(
            f"{technology:<22} & {investment:>10} & {lifetime:>4} & "
            f"{fom:>7} & {vom:>7} & {efficiency:>5} & "
            f"{fuel:>8} & {co2:>7} & {fixed:>10} \\\\"
        )


# =========================================================
# NETWORK COMPONENT EXPORTS
# =========================================================

def export_network_components(n: pypsa.Network) -> None:
    """
    Export and print all relevant component data from the solved network.
    """
    generator_columns = [
        "bus",
        "carrier",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
        "p_nom_min",
        "p_nom_max",
        "capital_cost",
        "marginal_cost",
        "efficiency",
        "p_min_pu",
        "ramp_limit_up",
        "ramp_limit_down",
    ]

    link_columns = [
        "bus0",
        "bus1",
        "bus2",
        "carrier",
        "p_nom",
        "p_nom_opt",
        "p_nom_extendable",
        "capital_cost",
        "marginal_cost",
        "efficiency",
        "efficiency2",
        "energy_to_power_ratio",
    ]

    store_columns = [
        "bus",
        "carrier",
        "e_nom",
        "e_nom_opt",
        "e_nom_extendable",
        "capital_cost",
        "marginal_cost",
        "standing_loss",
        "e_cyclic",
    ]

    load_columns = [
        "bus",
        "carrier",
        "p_set",
    ]

    carrier_columns = [
        "color",
        "co2_emissions",
    ]

    bus_columns = [
        "carrier",
    ]

    line_columns = [
        "bus0",
        "bus1",
        "carrier",
        "s_nom",
        "s_nom_opt",
        "s_nom_extendable",
        "capital_cost",
        "x",
        "r",
    ]

    save_and_print(
        safe_select(n.generators, generator_columns),
        "Generators",
    )

    save_and_print(
        safe_select(n.links, link_columns),
        "Links",
    )

    save_and_print(
        safe_select(n.stores, store_columns),
        "Stores",
    )

    save_and_print(
        safe_select(n.loads, load_columns),
        "Loads",
    )

    save_and_print(
        safe_select(n.buses, bus_columns),
        "Buses",
    )

    save_and_print(
        safe_select(n.carriers, carrier_columns),
        "Carriers",
    )

    save_and_print(
        safe_select(n.lines, line_columns),
        "Lines",
    )


def export_dispatch_summaries(n: pypsa.Network) -> None:
    """
    Export annual generation, link flows, store capacities and load summaries.
    """
    if not n.generators.empty:
        generator_summary = n.generators[["bus", "carrier", "p_nom_opt", "capital_cost", "marginal_cost", "efficiency"]].copy()
        generator_summary["annual_generation_MWh"] = n.generators_t.p.clip(lower=0.0).sum(axis=0)
        generator_summary["capacity_factor"] = (
            generator_summary["annual_generation_MWh"]
            / (generator_summary["p_nom_opt"].replace(0.0, pd.NA) * len(n.snapshots))
        )

        save_and_print(generator_summary, "Generator annual summary")

    if not n.links.empty:
        link_summary = safe_select(
            n.links,
            ["bus0", "bus1", "bus2", "carrier", "p_nom_opt", "capital_cost", "marginal_cost", "efficiency", "efficiency2"],
        )

        for column in ["p0", "p1", "p2"]:
            attr = getattr(n.links_t, column, None)

            if attr is not None and not attr.empty:
                link_summary[f"annual_{column}_positive_MWh"] = attr.clip(lower=0.0).sum(axis=0)
                link_summary[f"annual_{column}_negative_MWh"] = attr.clip(upper=0.0).sum(axis=0)

        save_and_print(link_summary, "Link annual summary")

    if not n.stores.empty:
        store_summary = safe_select(
            n.stores,
            ["bus", "carrier", "e_nom_opt", "capital_cost", "standing_loss", "e_cyclic"],
        )

        if not n.stores_t.e.empty:
            store_summary["max_state_of_charge_MWh"] = n.stores_t.e.max(axis=0)
            store_summary["min_state_of_charge_MWh"] = n.stores_t.e.min(axis=0)
            store_summary["mean_state_of_charge_MWh"] = n.stores_t.e.mean(axis=0)

        save_and_print(store_summary, "Store annual summary")

    if not n.loads.empty:
        load_summary = n.loads[["bus", "carrier"]].copy()
        load_summary["annual_demand_MWh"] = n.loads_t.p_set.sum(axis=0)
        load_summary["peak_demand_MW"] = n.loads_t.p_set.max(axis=0)
        load_summary["mean_demand_MW"] = n.loads_t.p_set.mean(axis=0)

        save_and_print(load_summary, "Load annual summary")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 260)

    print(f"\nLoading network:\n{NETWORK_FILE}")
    n = pypsa.Network(NETWORK_FILE)
    n.sanitize()

    print(f"\nLoading and preparing costs:\n{COST_FILE}")
    prepared_costs = prepare_costs(
        cost_file=COST_FILE,
        financial_parameters=FINANCIAL_PARAMETERS,
        number_of_years=FINANCIAL_PARAMETERS["nyears"],
    )

    print_section("NETWORK OBJECTIVE")
    print(n.objective)

    print_section("SNAPSHOTS")
    print(f"First snapshot: {n.snapshots[0]}")
    print(f"Last snapshot:  {n.snapshots[-1]}")
    print(f"Number of snapshots: {len(n.snapshots)}")

    export_network_components(n)
    export_dispatch_summaries(n)

    technology_table = build_technology_assumption_table(prepared_costs)

    print_section("TECHNO-ECONOMIC INPUT TABLE FROM PREPARED COSTS")
    print(
        technology_table.round(
            {
                "Investment [EUR/MW]": 1,
                "Lifetime [yr]": 1,
                "FOM [%/yr]": 4,
                "VOM [EUR/MWh]": 4,
                "Efficiency [-]": 3,
                "Fuel price [EUR/MWh_th]": 4,
                "CO2 intensity [tCO2/MWh_th]": 4,
                "Fixed [EUR/MW/yr]": 1,
            }
        ).to_string(index=False)
    )

    technology_table.to_csv(
        OUTPUT_DIR / "techno_economic_inputs_from_costs.csv",
        index=False,
    )

    print_latex_rows(technology_table)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()