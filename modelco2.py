"""
CO2 sensitivity analysis for the single-country PyPSA model.

Workflow
--------
1. Build and optimise an unconstrained baseline with the CO2 price included
   in marginal costs.
2. Calculate baseline annual CO2 emissions.
3. Re-run the model with CO2 caps set as fractions of the baseline emissions.
4. Store costs, emissions, capacities, battery sizes, and generation mix.
5. Plot the sensitivity results.

This script assumes:
- your model file is named `model.py`
- battery should remain active
- PyPSA's GlobalConstraint(type="primary_energy") is used for the CO2 cap
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from model import (
    prepare_costs,
    load_country_timeseries,
    create_regional_network,
    apply_battery_ratio_constraint,
)

OUTPUT_DIR = Path("results/co2_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs(
    financial_parameters: dict,
    scenario_parameters: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Load cost data and all country time series once.
    """
    cost_file = f"cost_data/costs_{financial_parameters['year']}.csv"
    timeseries_file = "Data/time_series_60min_singleindex_alldata.csv"

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

    return cost_data, all_timeseries_data


def build_and_optimize_network(
    cost_data: pd.DataFrame,
    all_timeseries_data: dict,
    scenario_parameters: dict,
    co2_price: float,
    co2_limit_mt: float | None = None,
) -> pypsa.Network:
    """
    Build and optimise a network with optional CO2 cap.
    """
    n = create_regional_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        with_battery_storage=scenario_parameters["with_battery_storage"],
        with_interconnectors=scenario_parameters["with_interconnectors"],
        co2_price=co2_price,
    )

    if co2_limit_mt is not None:
        n.add(
            "GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit_mt * 1e6,  # Mt -> tCO2
        )

    extra_functionality = (
        apply_battery_ratio_constraint
        if scenario_parameters["with_battery_storage"]
        else None
    )

    n.optimize(
        n.snapshots,
        extra_functionality=extra_functionality,
        solver_name="gurobi",
    )

    return n


def calculate_network_emissions(n: pypsa.Network) -> float:
    """
    Calculate annual CO2 emissions from generator dispatch in Mt CO2.

    Emissions are computed as:
        emissions = dispatch / efficiency * carrier_co2_intensity
    """
    if not hasattr(n, "objective") or n.objective is None:
        raise ValueError("Network has not been optimised yet.")

    total_emissions_tco2 = 0.0

    for gen in n.generators.index:
        if gen not in n.generators_t.p.columns:
            continue

        carrier = n.generators.at[gen, "carrier"]
        dispatch = n.generators_t.p[gen]
        efficiency = n.generators.at[gen, "efficiency"]

        if (
            carrier in n.carriers.index
            and "co2_emissions" in n.carriers.columns
            and pd.notna(efficiency)
            and efficiency > 0
        ):
            co2_intensity = n.carriers.at[carrier, "co2_emissions"]
            total_emissions_tco2 += (dispatch / efficiency * co2_intensity).sum()

    return total_emissions_tco2 / 1e6


def get_generation_by_carrier(n: pypsa.Network) -> dict:
    """
    Return total annual electricity generation by carrier in MWh.
    """
    generation = {}

    for gen in n.generators.index:
        if gen not in n.generators_t.p.columns:
            continue

        carrier = n.generators.at[gen, "carrier"]
        gen_mwh = n.generators_t.p[gen].sum()
        generation[carrier] = generation.get(carrier, 0.0) + gen_mwh

    for carrier in n.carriers.index:
        generation.setdefault(carrier, 0.0)

    return generation


def extract_result_row(
    n: pypsa.Network,
    scenario_name: str,
    co2_cap_fraction: float | float,
    co2_cap_mt: float | float,
    actual_emissions_mt: float,
) -> dict:
    """
    Extract one result row from an optimised network.
    """
    row = {
        "scenario": scenario_name,
        "co2_cap_fraction": co2_cap_fraction,
        "co2_cap_mt": co2_cap_mt,
        "system_cost_eur": n.objective,
        "actual_emissions_mt": actual_emissions_mt,
    }

    for gen in n.generators.index:
        row[f"capacity_{gen}"] = n.generators.at[gen, "p_nom_opt"]

    if "DK_battery_store" in n.stores.index:
        row["battery_energy_mwh"] = n.stores.at["DK_battery_store", "e_nom_opt"]
    if "DK_battery_charger" in n.links.index:
        row["battery_charger_mw"] = n.links.at["DK_battery_charger", "p_nom_opt"]
    if "DK_battery_discharger" in n.links.index:
        row["battery_discharger_mw"] = n.links.at["DK_battery_discharger", "p_nom_opt"]

    generation_by_carrier = get_generation_by_carrier(n)
    for carrier, gen_mwh in generation_by_carrier.items():
        row[f"generation_{carrier}"] = gen_mwh

    return row


def run_co2_sensitivity_analysis(
    financial_parameters: dict,
    scenario_parameters: dict,
    co2_cap_fractions: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run baseline + CO2 cap sensitivity analysis.
    """
    if co2_cap_fractions is None:
        co2_cap_fractions = [0.8, 0.6, 0.4, 0.2, 0.1]

    print("\n" + "=" * 80)
    print("CO2 SENSITIVITY ANALYSIS")
    print("=" * 80)

    cost_data, all_timeseries_data = load_inputs(
        financial_parameters=financial_parameters,
        scenario_parameters=scenario_parameters,
    )

    results = []
    co2_price = financial_parameters["co2_price"]

    print("\n" + "-" * 80)
    print("BASELINE: Running optimisation with NO CO2 cap...")
    print("-" * 80)

    n_baseline = build_and_optimize_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        scenario_parameters=scenario_parameters,
        co2_price=co2_price,
        co2_limit_mt=None,
    )

    baseline_emissions = calculate_network_emissions(n_baseline)

    print(f"Baseline system cost: €{n_baseline.objective:,.0f}")
    print(f"Baseline annual emissions: {baseline_emissions:.3f} Mt CO2")

    results.append(
        extract_result_row(
            n=n_baseline,
            scenario_name="baseline",
            co2_cap_fraction=np.nan,
            co2_cap_mt=np.nan,
            actual_emissions_mt=baseline_emissions,
        )
    )

    print("\n" + "-" * 80)
    print(f"Running {len(co2_cap_fractions)} CO2 cap scenarios...")
    print("-" * 80)

    for i, fraction in enumerate(sorted(co2_cap_fractions, reverse=True), start=1):
        co2_cap_mt = baseline_emissions * fraction

        print(
            f"\n[{i}/{len(co2_cap_fractions)}] "
            f"CO2 cap = {fraction:.0%} of baseline = {co2_cap_mt:.3f} Mt CO2"
        )

        n_scenario = build_and_optimize_network(
            cost_data=cost_data,
            all_timeseries_data=all_timeseries_data,
            scenario_parameters=scenario_parameters,
            co2_price=co2_price,
            co2_limit_mt=co2_cap_mt,
        )

        if n_scenario.objective is None or np.isnan(n_scenario.objective):
            print(f"  Optimization failed for cap {co2_cap_mt:.3f} Mt CO2")
            continue

        actual_emissions = calculate_network_emissions(n_scenario)
        cost_change_pct = (
            (n_scenario.objective - n_baseline.objective) / n_baseline.objective * 100
        )

        print(f"  System cost: €{n_scenario.objective:,.0f} ({cost_change_pct:+.1f}%)")
        print(f"  Actual emissions: {actual_emissions:.3f} Mt CO2")

        results.append(
            extract_result_row(
                n=n_scenario,
                scenario_name=f"cap_{int(fraction * 100)}pct",
                co2_cap_fraction=fraction,
                co2_cap_mt=co2_cap_mt,
                actual_emissions_mt=actual_emissions,
            )
        )

    return pd.DataFrame(results)


def plot_co2_sensitivity(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create plots for the CO2 sensitivity analysis.
    """
    plot_df = results_df[results_df["co2_cap_fraction"].notna()].copy()
    plot_df = plot_df.sort_values("co2_cap_mt")

    baseline = results_df[results_df["scenario"] == "baseline"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Plot 1: Generation mix
    ax = axes[0, 0]
    x = plot_df["co2_cap_mt"].values

    carriers_to_plot = ["solar", "onwind", "offwind", "gas", "coal", "nuclear"]
    colors = {
        "solar": "#EBCB3B",
        "onwind": "#5AA469",
        "offwind": "#2E86AB",
        "gas": "#D08770",
        "coal": "#5C5C5C",
        "nuclear": "#8F6BB3",
    }

    bottom = np.zeros(len(plot_df))
    for carrier in carriers_to_plot:
        col = f"generation_{carrier}"
        if col in plot_df.columns:
            values = plot_df[col].fillna(0.0).values / 1000.0
            ax.fill_between(
                x,
                bottom,
                bottom + values,
                label=carrier,
                alpha=0.7,
                color=colors[carrier],
            )
            bottom += values

    ax.set_xlabel("CO2 cap [Mt CO2/year]")
    ax.set_ylabel("Annual generation [GWh]")
    ax.set_title("Generation mix vs CO2 cap")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Plot 2: System cost
    ax = axes[0, 1]
    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["system_cost_eur"] / 1e9,
        marker="o",
        linewidth=2,
        markersize=6,
        color="red",
    )
    ax.axhline(
        y=baseline["system_cost_eur"] / 1e9,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Baseline: €{baseline['system_cost_eur']/1e9:.2f}B",
    )
    ax.set_xlabel("CO2 cap [Mt CO2/year]")
    ax.set_ylabel("Annual system cost [€ billion]")
    ax.set_title("System cost vs CO2 cap")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Plot 3: Actual emissions vs cap
    ax = axes[1, 0]
    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["actual_emissions_mt"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="Actual emissions",
        color="blue",
    )
    ax.plot(
        plot_df["co2_cap_mt"],
        plot_df["co2_cap_mt"],
        linestyle="--",
        color="red",
        alpha=0.7,
        label="Cap",
    )
    ax.axhline(
        y=baseline["actual_emissions_mt"],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Baseline: {baseline['actual_emissions_mt']:.3f} Mt CO2",
    )
    ax.set_xlabel("CO2 cap [Mt CO2/year]")
    ax.set_ylabel("Emissions [Mt CO2/year]")
    ax.set_title("Actual emissions vs CO2 cap")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Plot 4: Capacity mix
    ax = axes[1, 1]
    capacity_cols = [
        col
        for col in plot_df.columns
        if col.startswith("capacity_")
        and any(key in col for key in ["CCGT", "coal", "solar", "wind", "nuclear"])
    ]

    for col in sorted(capacity_cols):
        label = col.replace("capacity_", "")
        ax.plot(
            plot_df["co2_cap_mt"],
            plot_df[col],
            marker="o",
            linewidth=2,
            label=label,
        )

    if "battery_energy_mwh" in plot_df.columns:
        ax.plot(
            plot_df["co2_cap_mt"],
            plot_df["battery_energy_mwh"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="battery_energy_mwh",
        )

    ax.set_xlabel("CO2 cap [Mt CO2/year]")
    ax.set_ylabel("Installed capacity [MW or MWh]")
    ax.set_title("Capacities vs CO2 cap")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.savefig(output_dir / "co2_sensitivity_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_dir / 'co2_sensitivity_analysis.png'}")


def main() -> None:
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

    co2_cap_fractions = [0.8, 0.6, 0.4, 0.2, 0.1]

    results_df = run_co2_sensitivity_analysis(
        financial_parameters=financial_parameters,
        scenario_parameters=scenario_parameters,
        co2_cap_fractions=co2_cap_fractions,
    )

    results_path = OUTPUT_DIR / "co2_sensitivity_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary_cols = [
        "scenario",
        "co2_cap_fraction",
        "co2_cap_mt",
        "system_cost_eur",
        "actual_emissions_mt",
    ]
    print(results_df[summary_cols].to_string(index=False))

    print("\nCreating plots...")
    plot_co2_sensitivity(results_df, OUTPUT_DIR)

    print(f"\nAnalysis complete. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()