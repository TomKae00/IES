from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

# Ensure the project root is on the import path when running from analysis/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.helpers import (
    prepare_costs,
    load_country_timeseries,
    calculate_conventional_marginal_cost,
)
from model.model import (
    create_network,
    custom_constraints,
)


OUTPUT_DIR = Path("results/co2_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def optimize_network(n: pypsa.Network, scenario: dict):
    def extra_functionality(network: pypsa.Network, snapshots):
        custom_constraints(network, snapshots, scenario)

    status = n.optimize(
        n.snapshots,
        extra_functionality=extra_functionality,
        solver_name="gurobi",
        solver_options={},
        include_objective_constant=False,
    )
    return status


def calculate_network_emissions(n: pypsa.Network) -> float:
    """
    Return annual CO2 emissions in Mt CO2.

    Uses:
    generator dispatch [MWh_el]
    / efficiency
    * carrier co2_emissions [tCO2/MWh_th]
    """
    if n.objective is None:
        raise ValueError("Network has not been optimized.")

    total_tco2 = 0.0

    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]

        if gen not in n.generators_t.p.columns:
            continue

        if carrier not in n.carriers.index:
            continue

        co2_intensity = n.carriers.at[carrier, "co2_emissions"] \
            if "co2_emissions" in n.carriers.columns else 0.0

        efficiency = n.generators.at[gen, "efficiency"]

        if pd.isna(efficiency) or efficiency <= 0:
            efficiency = 1.0

        dispatch_mwh = n.generators_t.p[gen].clip(lower=0)
        primary_energy_mwh = dispatch_mwh / efficiency
        total_tco2 += (primary_energy_mwh * co2_intensity).sum()

    return total_tco2 / 1e6


def get_generation_by_carrier(n: pypsa.Network) -> dict:
    generation = {}

    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]

        if gen in n.generators_t.p.columns:
            generation[carrier] = generation.get(carrier, 0.0) + n.generators_t.p[gen].clip(lower=0).sum()

    return generation


def get_capacity_by_carrier(n: pypsa.Network) -> dict:
    capacities = {}

    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]
        cap = n.generators.at[gen, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else n.generators.at[gen, "p_nom"]
        capacities[carrier] = capacities.get(carrier, 0.0) + cap

    return capacities


def get_battery_capacities(n: pypsa.Network) -> dict:
    """
    Flexible battery extraction.
    Works even when names are no longer DK_battery_store, etc.
    """
    result = {
        "battery_energy_mwh": 0.0,
        "battery_charger_mw": 0.0,
        "battery_discharger_mw": 0.0,
    }

    if len(n.stores) > 0:
        battery_stores = n.stores[
            n.stores.index.str.contains("battery", case=False, regex=False)
            | n.stores.carrier.astype(str).str.contains("battery", case=False, regex=False)
        ]

        if not battery_stores.empty:
            if "e_nom_opt" in battery_stores.columns:
                result["battery_energy_mwh"] = battery_stores["e_nom_opt"].sum()
            else:
                result["battery_energy_mwh"] = battery_stores["e_nom"].sum()

    if len(n.links) > 0:
        battery_links = n.links[
            n.links.index.str.contains("battery", case=False, regex=False)
            | n.links.carrier.astype(str).str.contains("battery", case=False, regex=False)
        ]

        chargers = battery_links[
            battery_links.index.str.contains("charger", case=False, regex=False)
        ]

        dischargers = battery_links[
            battery_links.index.str.contains("discharger", case=False, regex=False)
        ]

        if not chargers.empty:
            result["battery_charger_mw"] = chargers["p_nom_opt"].sum()

        if not dischargers.empty:
            result["battery_discharger_mw"] = dischargers["p_nom_opt"].sum()

    return result


def build_result_entry(
    n: pypsa.Network,
    co2_cap_fraction: float,
    co2_cap_mt: float,
    actual_emissions_mt: float,
) -> dict:

    result = {
        "co2_cap_fraction": co2_cap_fraction,
        "co2_cap_mt": co2_cap_mt,
        "system_cost_eur": n.objective,
        "actual_emissions_mt": actual_emissions_mt,
    }

    for carrier, cap in get_capacity_by_carrier(n).items():
        result[f"capacity_{carrier}_mw"] = cap

    for carrier, gen in get_generation_by_carrier(n).items():
        result[f"generation_{carrier}_mwh"] = gen

    result.update(get_battery_capacities(n))

    return result


def add_co2_constraint(n: pypsa.Network, co2_cap_mt: float):
    n.add(
        "GlobalConstraint",
        "co2_limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_cap_mt * 1e6,
    )

def carrier_order(columns):
    preferred_order = [
        "nuclear",
        "coal",
        "gas CCGT",
        "offwind",
        "onwind",
        "solar",
    ]

    return [c for c in preferred_order if c in columns] + [
        c for c in columns if c not in preferred_order
    ]


def plot_weekly_dispatch(
    n: pypsa.Network,
    week_start: str = "2016-01-01",
    title: str = "Weekly dispatch",
    output_dir: Path = OUTPUT_DIR,
):
    week_start = pd.to_datetime(week_start)
    week_end = week_start + pd.Timedelta(days=7)

    week_snapshots = n.snapshots[
        (n.snapshots >= week_start) & (n.snapshots < week_end)
    ]

    if len(week_snapshots) == 0:
        print(f"No snapshots found for week starting {week_start}")
        return

    dispatch = n.generators_t.p.loc[week_snapshots].copy()

    carrier_dispatch = pd.DataFrame(index=dispatch.index)

    for carrier in n.generators.carrier.unique():
        gens = n.generators.index[n.generators.carrier == carrier]

        available_gens = [g for g in gens if g in dispatch.columns]

        if available_gens:
            carrier_dispatch[carrier] = dispatch[available_gens].clip(lower=0).sum(axis=1)

    carrier_dispatch = carrier_dispatch.loc[:, carrier_dispatch.sum() > 0]
    carrier_dispatch = carrier_dispatch[carrier_order(carrier_dispatch.columns)]

    if carrier_dispatch.empty:
        print("No dispatch data to plot.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    carrier_dispatch.plot.area(ax=ax, stacked=True, linewidth=0)

    ax.set_title(title)
    ax.set_ylabel("Dispatch [MW]")
    ax.set_xlabel("")
    ax.grid(alpha=0.3)
    ax.legend(title="Carrier", ncol=3, fontsize=9)

    plt.tight_layout()

    filename = title.lower().replace(" ", "_").replace("%", "pct").replace(".", "") + ".png"
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved dispatch plot to {output_dir / filename}")


def run_co2_sensitivity_analysis(
    cost_file: str,
    timeseries_file: str,
    financial_parameters: dict,
    scenario_parameters: dict,
    co2_cap_fractions: list[float] | None = None,
    co2_price: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pypsa.Network]:

    if co2_cap_fractions is None:
        co2_cap_fractions=[
    0.8,  # 20% reduction
    0.6,  # 40% reduction
    0.4,  # 60% reduction
    0.3,  # 70% reduction, Denmark target
    0.2,  # 80% reduction
    0.1,  # 90% reduction
]

    print("\n" + "=" * 80)
    print("CO2 SENSITIVITY ANALYSIS")
    print("=" * 80)

    cost_data = prepare_costs(
        cost_file=cost_file,
        financial_parameters=financial_parameters,
        number_of_years=financial_parameters["nyears"],
    )

    all_timeseries_data = {}

    for country_code in scenario_parameters["countries"]:
        all_timeseries_data[country_code] = load_country_timeseries(
            timeseries_file=timeseries_file,
            country_code=country_code,
            year=scenario_parameters["weather_year"],
        )

    scenario = {
        "name": scenario_parameters.get("name", "co2_sensitivity"),
        "weather_year": scenario_parameters["weather_year"],
        "countries": scenario_parameters["countries"],
        "with_battery_storage": scenario_parameters.get("with_battery_storage", True),
        "with_interconnectors": scenario_parameters.get("with_interconnectors", False),
        "with_ch4_network": scenario_parameters.get("with_ch4_network", False),
        "with_h2_network": scenario_parameters.get("with_h2_network", False),
        "with_heat_sector": scenario_parameters.get("with_heat_sector", False),
        "with_heat_storage": scenario_parameters.get("with_heat_storage", False),
        "co2_price": co2_price,
        "co2_limit": None,
    }

    results = []

    print("\nRunning baseline with no CO2 cap...")

    n_baseline = create_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        scenario=scenario,
        heat_timeseries=None,
    )

    optimize_network(n_baseline, scenario)

    baseline_emissions = calculate_network_emissions(n_baseline)

    print(f"Baseline cost: €{n_baseline.objective:,.0f}")
    print(f"Baseline emissions: {baseline_emissions:.3f} Mt CO2")

    results.append(
        build_result_entry(
            n=n_baseline,
            co2_cap_fraction=1.0,
            co2_cap_mt=np.inf,
            actual_emissions_mt=baseline_emissions,
        )
    )

    plot_weekly_dispatch(
        n_baseline,
        week_start=f"{scenario_parameters['weather_year']}-01-01",
        title="Baseline weekly dispatch",
        output_dir=OUTPUT_DIR,
    )

    for i, fraction in enumerate(sorted(co2_cap_fractions, reverse=True), start=1):
        co2_cap_mt = baseline_emissions * fraction

        print("\n" + "-" * 80)
        print(f"[{i}/{len(co2_cap_fractions)}] CO2 cap: {fraction:.0%} of baseline")
        print(f"Cap: {co2_cap_mt:.3f} Mt CO2")
        print("-" * 80)

        scenario_with_cap = scenario.copy()
        scenario_with_cap["co2_limit"] = co2_cap_mt * 1e6

        n_scenario = create_network(
            cost_data=cost_data,
            all_timeseries_data=all_timeseries_data,
            scenario=scenario_with_cap,
            heat_timeseries=None,
        )

        optimize_network(n_scenario, scenario_with_cap)

        if n_scenario.objective is None or pd.isna(n_scenario.objective):
            print(f"Optimization failed or infeasible for CO2 cap {co2_cap_mt:.3f} Mt")
            continue

        actual_emissions = calculate_network_emissions(n_scenario)

        cost_increase_pct = (
            (n_scenario.objective - n_baseline.objective)
            / n_baseline.objective
            * 100
        )

        print(f"System cost: €{n_scenario.objective:,.0f}")
        print(f"Cost increase: {cost_increase_pct:+.1f}%")
        print(f"Actual emissions: {actual_emissions:.3f} Mt CO2")

        results.append(
            build_result_entry(
                n=n_scenario,
                co2_cap_fraction=fraction,
                co2_cap_mt=co2_cap_mt,
                actual_emissions_mt=actual_emissions,
            )
        )

        if np.isclose(fraction, 0.30):
            plot_weekly_dispatch(
                n_scenario,
                week_start=f"{scenario_parameters['weather_year']}-01-01",
                title="Weekly dispatch 70% CO2 reduction",
                output_dir=OUTPUT_DIR,
            )

    results_df = pd.DataFrame(results)
    return results_df, cost_data, n_baseline


def plot_co2_sensitivity(results_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR):
    plot_df = results_df[np.isfinite(results_df["co2_cap_mt"])].copy()

    # Add reduction labels
    plot_df["co2_reduction_pct"] = (1 - plot_df["co2_cap_fraction"]) * 100

    plot_df = plot_df.sort_values("co2_cap_mt")

    baseline = results_df[results_df["co2_cap_fraction"] == 1.0].iloc[0]

    target_fraction = 0.30   # 70% reduction means 30% of baseline emissions remain
    target_cap_mt = baseline["actual_emissions_mt"] * target_fraction

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    def format_x_axis(ax):
        ax.set_xlabel("CO2 cap [Mt CO2/year]\nReduction from baseline [%]")
        ticks = plot_df["co2_cap_mt"].values
        labels = [
            f"{cap:.2f}\n{red:.0f}%"
            for cap, red in zip(plot_df["co2_cap_mt"], plot_df["co2_reduction_pct"])
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=0)

        # 70% reduction target line
        ax.axvline(
            target_cap_mt,
            linestyle=":",
            linewidth=2,
            label="70% reduction target",
        )

    # -----------------------------------------------------
    # Generation mix
    # -----------------------------------------------------
    ax = axes[0, 0]

    generation_cols = [
        c for c in plot_df.columns
        if c.startswith("generation_") and c.endswith("_mwh")
    ]

    bottom = np.zeros(len(plot_df))

    for col in sorted(generation_cols):
        carrier = col.replace("generation_", "").replace("_mwh", "")
        values = plot_df[col].fillna(0).values / 1000

        ax.fill_between(
            plot_df["co2_cap_mt"],
            bottom,
            bottom + values,
            alpha=0.75,
            label=carrier,
        )

        bottom += values

    ax.set_title("Generation mix vs CO2 cap")
    ax.set_ylabel("Generation [GWh/year]")
    ax.grid(alpha=0.25)
    format_x_axis(ax)
    ax.legend(fontsize=8)

    # -----------------------------------------------------
    # System cost
    # -----------------------------------------------------
    ax = axes[0, 1]

    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["system_cost_eur"] / 1e9,
        marker="o",
        linewidth=2,
    )

    ax.axhline(
        baseline["system_cost_eur"] / 1e9,
        linestyle="--",
        alpha=0.6,
        label="Baseline",
    )

    ax.set_title("System cost vs CO2 cap")
    ax.set_ylabel("System cost [€bn/year]")
    ax.grid(alpha=0.25)
    format_x_axis(ax)
    ax.legend()

    # -----------------------------------------------------
    # Actual emissions
    # -----------------------------------------------------
    ax = axes[1, 0]

    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["actual_emissions_mt"],
        marker="o",
        linewidth=2,
        label="Actual emissions",
    )

    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["co2_cap_mt"],
        linestyle="--",
        label="CO2 cap",
    )

    ax.axhline(
        baseline["actual_emissions_mt"],
        linestyle=":",
        alpha=0.6,
        label="Baseline",
    )

    ax.set_title("Actual emissions vs CO2 cap")
    ax.set_ylabel("Emissions [Mt CO2/year]")
    ax.grid(alpha=0.25)
    format_x_axis(ax)
    ax.legend()

    # -----------------------------------------------------
    # Capacity mix
    # -----------------------------------------------------
    ax = axes[1, 1]

    capacity_cols = [
        c for c in plot_df.columns
        if c.startswith("capacity_") and c.endswith("_mw")
    ]

    for col in sorted(capacity_cols):
        carrier = col.replace("capacity_", "").replace("_mw", "")

        ax.plot(
            plot_df["co2_cap_mt"],
            plot_df[col].fillna(0),
            marker="o",
            linewidth=2,
            label=carrier,
        )

    if "battery_energy_mwh" in plot_df.columns:
        ax.plot(
            plot_df["co2_cap_mt"],
            plot_df["battery_energy_mwh"].fillna(0),
            marker="s",
            linewidth=2,
            label="battery energy [MWh]",
        )

    ax.set_title("Capacity mix vs CO2 cap")
    ax.set_ylabel("Capacity [MW] / Battery energy [MWh]")
    ax.grid(alpha=0.25)
    format_x_axis(ax)
    ax.legend(fontsize=8)

    fig.savefig(
        output_dir / "co2_sensitivity_analysis.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot to {output_dir / 'co2_sensitivity_analysis.png'}")


def print_summary(results_df: pd.DataFrame, cost_data: pd.DataFrame, n_baseline: pypsa.Network, co2_price: float):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # -----------------------------------------------------
    # Identify baseline and 70% reduction
    # -----------------------------------------------------
    baseline_row = results_df[
        results_df["co2_cap_fraction"] == 1.0
    ].iloc[0]

    target_fraction = 0.30  # 70% reduction
    target_row = results_df[
        np.isclose(results_df["co2_cap_fraction"], target_fraction)
    ].iloc[0]

    # -----------------------------------------------------
    # CO2
    # -----------------------------------------------------
    print("\nCO2 EMISSIONS:")
    print(f"Baseline: {baseline_row['actual_emissions_mt']:.3f} Mt CO2")
    print(f"70% reduction: {target_row['actual_emissions_mt']:.3f} Mt CO2")

    # -----------------------------------------------------
    # CAPACITY MIX FUNCTION
    # -----------------------------------------------------
    def print_capacity_mix(row, title):
        print(f"\n{title}")
        print("-" * 40)

        capacity_cols = [
            c for c in row.index if c.startswith("capacity_") and c.endswith("_mw")
        ]

        total_capacity = sum(row[c] for c in capacity_cols if not pd.isna(row[c]))

        for col in sorted(capacity_cols):
            carrier = col.replace("capacity_", "").replace("_mw", "")
            value = row[col] if not pd.isna(row[col]) else 0.0
            share = value / total_capacity * 100 if total_capacity > 0 else 0.0

            print(f"{carrier}: {value:.1f} MW ({share:.1f}%)")

        # 🔋 ADD BATTERY HERE (THIS IS THE KEY PART)
        print("\nBattery:")
        print(f"Energy: {row.get('battery_energy_mwh', 0.0):.1f} MWh")
        print(f"Charging: {row.get('battery_charger_mw', 0.0):.1f} MW")
        print(f"Discharging: {row.get('battery_discharger_mw', 0.0):.1f} MW")

    # -----------------------------------------------------
    # PRINT BOTH CASES
    # -----------------------------------------------------
    print_capacity_mix(baseline_row, "Baseline capacity by carrier:")
    print_capacity_mix(target_row, "Capacity at 70% CO2 reduction:")


def main():
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
        "countries": ["DK"],
    }

    file_paths = {
        "cost_file": f"Data/costs_{financial_parameters['year']}.csv",
        "timeseries_file": "Data\\time_series_60min_singleindex_filtered_2015-2020.csv",
    }

    results_df, cost_data, n_baseline = run_co2_sensitivity_analysis(
        cost_file=file_paths["cost_file"],
        timeseries_file=file_paths["timeseries_file"],
        financial_parameters=financial_parameters,
        scenario_parameters=scenario_parameters,
        co2_cap_fractions=[0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
        co2_price=financial_parameters["co2_price"],
    )

    results_df.to_csv(OUTPUT_DIR / "co2_sensitivity_results.csv", index=False)

    print(f"\nSaved results to {OUTPUT_DIR / 'co2_sensitivity_results.csv'}")

    plot_co2_sensitivity(results_df, OUTPUT_DIR)

    print_summary(
        results_df=results_df,
        cost_data=cost_data,
        n_baseline=n_baseline,
        co2_price=financial_parameters["co2_price"],
    )

    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()