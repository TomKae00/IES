"""
CO2 Sensitivity Analysis: Investigate optimal capacity mix under global CO2 constraints.

Approach:
1. Run baseline optimization with no CO2 cap
2. Calculate baseline annual CO2 emissions
3. Create scenarios with CO2 caps at fractions of baseline: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
4. Run optimization for each scenario
5. Track capacities, generation mix, costs, and actual emissions
6. Create plots showing sensitivities

Uses PyPSA's GlobalConstraint with type="primary_energy" and carrier_attribute="co2_emissions".
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa

from model import (
    prepare_costs,
    load_country_timeseries,
    create_regional_network,
    apply_battery_ratio_constraint,
    calculate_conventional_marginal_cost,
)


OUTPUT_DIR = Path("results/co2_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_network_emissions(n: pypsa.Network) -> float:
    """
    Calculate annual CO2 emissions from an optimized network in Mt CO2.
    
    Uses the carrier co2_emissions attribute and generator/link dispatch.
    
    Formula: emission_tco2 = sum_g,t [ (co2_intensity_g / efficiency_g) * dispatch_g,t ]
    
    Parameters
    ----------
    n : pypsa.Network
        Optimized network with solution.
    
    Returns
    -------
    float
        Annual CO2 emissions in Mt CO2.
    """
    if not hasattr(n, 'objective') or n.objective is None:
        raise ValueError("Network has not been optimized yet. No solution to calculate emissions from.")
    
    total_emissions_tco2 = 0.0
    
    # Get CO2 emissions from generators
    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]
        if gen in n.generators_t.p.columns:
            dispatch = n.generators_t.p[gen]  # MWh for each snapshot
            efficiency = n.generators.at[gen, "efficiency"]
            
            # Get carrier CO2 intensity
            if carrier in n.carriers.index and "co2_emissions" in n.carriers.columns:
                co2_intensity = n.carriers.at[carrier, "co2_emissions"]
            else:
                co2_intensity = 0.0
            
            # Primary energy = electrical energy / efficiency
            # Emissions = primary energy * co2_intensity_per_MWh_th
            primary_energy = dispatch / efficiency if efficiency > 0 else 0
            emissions = (primary_energy * co2_intensity).sum()
            total_emissions_tco2 += emissions
    
    # Get CO2 emissions from links (chargers/dischargers if applicable)
    for link in n.links.index:
        carrier = n.links.at[link, "carrier"]
        if link in n.links_t.p0.columns:
            dispatch = n.links_t.p0[link]  # MWh from bus0 to bus1
            efficiency = n.links.at[link, "efficiency"]
            
            if carrier in n.carriers.index and "co2_emissions" in n.carriers.columns:
                co2_intensity = n.carriers.at[carrier, "co2_emissions"]
            else:
                co2_intensity = 0.0
            
            # For links, dispatch is directional; count absolute value to account for input energy
            primary_energy = dispatch.abs() / efficiency if efficiency > 0 else 0
            emissions = (primary_energy * co2_intensity).sum()
            total_emissions_tco2 += emissions
    
    return total_emissions_tco2 / 1e6  # Convert tCO2 to Mt CO2


def plot_weekly_dispatch(n: pypsa.Network, week_start: str = '2016-01-01', title: str = 'Weekly Dispatch', output_dir: Path = None):
    """
    Plot the dispatch for the first week.
    
    Parameters
    ----------
    n : pypsa.Network
        Optimized network.
    week_start : str
        Start date of the week.
    title : str
        Plot title.
    output_dir : Path, optional
        Directory to save the plot.
    """
    week_end = pd.to_datetime(week_start) + pd.Timedelta(days=7)
    week_snapshots = n.snapshots[(n.snapshots >= week_start) & (n.snapshots < week_end)]
    
    # Select key generators in stacking order (nuclear at bottom)
    stacking_order = ['nuclear', 'coal', 'CCGT', 'solar', 'onshore_wind', 'offshore_wind']
    key_gens = []
    for tech in stacking_order:
        matching_gens = [gen for gen in n.generators.index if tech in gen]
        key_gens.extend(matching_gens)
    
    if not key_gens:
        return
    
    dispatch_data = n.generators_t.p.loc[week_snapshots, key_gens]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    dispatch_data.plot.area(ax=ax, stacked=True, linewidth=0)
    ax.set_title(title)
    ax.set_ylabel("Dispatch [MW]")
    ax.set_xlabel("")
    ax.legend(title="Technology", ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if output_dir:
        filename = title.replace(' ', '_').replace('%', 'pct').replace('.', '') + '.png'
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        print(f"Saved dispatch plot to {output_dir / filename}")
    plt.show()
    plt.close(fig)


def get_generation_by_carrier(n: pypsa.Network) -> dict:
    """
    Get total annual generation by carrier type.
    
    Parameters
    ----------
    n : pypsa.Network
        Optimized network.
    
    Returns
    -------
    dict
        Generation in MWh by carrier.
    """
    generation = {}
    
    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]
        if gen in n.generators_t.p.columns:
            gen_mwh = n.generators_t.p[gen].sum()
            generation[carrier] = generation.get(carrier, 0.0) + gen_mwh
    
    # Fill in missing carriers with zero
    for carrier in n.carriers.index:
        generation.setdefault(carrier, 0.0)
    
    return generation


def optimize_network(n: pypsa.Network) -> None:
    """
    Optimize a network with the battery ratio constraint applied.
    """
    n.optimize(
        n.snapshots,
        solver_name="gurobi",
        extra_functionality=apply_battery_ratio_constraint,
    )


def build_result_entry(
    n: pypsa.Network,
    co2_cap_fraction: float,
    co2_cap_mt: float,
    actual_emissions_mt: float,
) -> dict:
    """
    Build a result dictionary from an optimized network.
    """
    result = {
        "co2_cap_fraction": co2_cap_fraction,
        "co2_cap_mt": co2_cap_mt,
        "system_cost_eur": n.objective,
        "actual_emissions_mt": actual_emissions_mt,
    }

    for gen in n.generators.index:
        result[f"capacity_{gen}"] = n.generators.at[gen, "p_nom_opt"]

    if "DK_battery_store" in n.stores.index:
        result["battery_energy_mwh"] = n.stores.at["DK_battery_store", "e_nom_opt"]
    if "DK_battery_charger" in n.links.index:
        result["battery_charger_mw"] = n.links.at["DK_battery_charger", "p_nom_opt"]
    if "DK_battery_discharger" in n.links.index:
        result["battery_discharger_mw"] = n.links.at["DK_battery_discharger", "p_nom_opt"]

    for carrier, gen_mwh in get_generation_by_carrier(n).items():
        result[f"generation_{carrier}"] = gen_mwh

    return result


def run_co2_sensitivity_analysis(
    cost_file: str,
    timeseries_file: str,
    financial_parameters: dict,
    scenario_parameters: dict,
    co2_cap_fractions: list = None,
    co2_price: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pypsa.Network]:
    """
    Run sensitivity analysis across multiple CO2 cap scenarios.
    
    Returns
    -------
    tuple
        Results DataFrame, cost_data DataFrame, baseline network.
    """
    """
    Run sensitivity analysis across multiple CO2 cap scenarios.
    
    Parameters
    ----------
    cost_file : str
        Path to cost data CSV.
    timeseries_file : str
        Path to timeseries data CSV.
    financial_parameters : dict
        Financial parameters.
    scenario_parameters : dict
        Scenario parameters (weather_year, countries, etc.).
    co2_cap_fractions : list, optional
        Fractions of baseline emissions to use as caps.
        Default: [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    
    Returns
    -------
    tuple
        Results DataFrame, cost_data DataFrame, baseline network.
    """
    if co2_cap_fractions is None:
        co2_cap_fractions = [0.8, 0.5, 0.3, 0.2, 0.1]
    
    print("\n" + "=" * 80)
    print("CO2 SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    # Load cost and timeseries data once
    cost_data = prepare_costs(
        cost_file=cost_file,
        financial_parameters=financial_parameters,
        number_of_years=financial_parameters["nyears"],
    )
    
    # Print generator cost table in table that goes into results

    
    
    all_timeseries_data = {}
    for country_code in scenario_parameters["countries"]:
        all_timeseries_data[country_code] = load_country_timeseries(
            timeseries_file=timeseries_file,
            country_code=country_code,
            year=scenario_parameters["weather_year"],
        )
    
    results = []
    
    # Step 1: Run baseline optimization (no CO2 cap)
    print("\n" + "-" * 80)
    print("BASELINE: Running optimization with NO CO2 cap...")
    print("-" * 80)
    
    n_baseline = create_regional_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        with_battery_storage=scenario_parameters.get("with_battery_storage", True),
        with_interconnectors=scenario_parameters.get("with_interconnectors", False),
        co2_price=co2_price,
    )
    
    # Optimize without CO2 constraint
    n_baseline.optimize(
        n_baseline.snapshots,
        solver_name="gurobi",
    )
    
    baseline_emissions = calculate_network_emissions(n_baseline)
    print(f"Baseline system cost: €{n_baseline.objective:,.0f}")
    print(f"Baseline annual emissions: {baseline_emissions:.3f} Mt CO2")
    print(f"Carrier CO2 intensities (tCO2/MWh_th):")
    print(n_baseline.carriers[["co2_emissions"]])
    
    # Print capacity shares
    
    
    # Print weekly dispatch for baseline
    
    plot_weekly_dispatch(n_baseline, title='Baseline Weekly Dispatch', output_dir=Path("results/co2_sensitivity"))
    
    # Store baseline result
    gen_by_carrier = get_generation_by_carrier(n_baseline)
    result = {
        "co2_cap_fraction": 1.0,
        "co2_cap_mt": np.inf,
        "system_cost_eur": n_baseline.objective,
        "actual_emissions_mt": baseline_emissions,
    }
    
    # Add generator capacities
    for gen in n_baseline.generators.index:
        result[f"capacity_{gen}"] = n_baseline.generators.at[gen, "p_nom_opt"]
    
    # Add battery capacities if present
    if "DK_battery_store" in n_baseline.stores.index:
        result["battery_energy_mwh"] = n_baseline.stores.at["DK_battery_store", "e_nom_opt"]
    if "DK_battery_charger" in n_baseline.links.index:
        result["battery_charger_mw"] = n_baseline.links.at["DK_battery_charger", "p_nom_opt"]
    if "DK_battery_discharger" in n_baseline.links.index:
        result["battery_discharger_mw"] = n_baseline.links.at["DK_battery_discharger", "p_nom_opt"]
    
    # Add generation by carrier
    for carrier, gen_mwh in gen_by_carrier.items():
        result[f"generation_{carrier}"] = gen_mwh
    
    results.append(result)
    
    # Step 2: Run scenarios with CO2 caps at fractions of baseline
    print("\n" + "-" * 80)
    print(f"Running {len(co2_cap_fractions)} scenarios with CO2 caps as fractions of baseline...")
    print("-" * 80)
    
    for i, fraction in enumerate(sorted(co2_cap_fractions, reverse=True)):
        co2_cap_mt = baseline_emissions * fraction
        
        print(f"\n[{i+1}/{len(co2_cap_fractions)}] CO2 Cap Scenario: {fraction:.1%} of baseline ({co2_cap_mt:.3f} Mt CO2)")
        
        # Create new network for this scenario
        n_scenario = create_regional_network(
            cost_data=cost_data,
            all_timeseries_data=all_timeseries_data,
            with_battery_storage=scenario_parameters.get("with_battery_storage", True),
            with_interconnectors=scenario_parameters.get("with_interconnectors", False),
            co2_price=co2_price,
        )
        
        # Add global CO2 constraint
        n_scenario.add(
            "GlobalConstraint",
            "co2_limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_cap_mt * 1e6,  # Convert Mt to tCO2
        )
        
        # Optimize with CO2 constraint
        n_scenario.optimize(
            n_scenario.snapshots,
            solver_name="gurobi",
        )
        
        # Check if optimization was successful
        if n_scenario.objective is None or np.isnan(n_scenario.objective):
            print(f"  ⚠ Optimization failed or infeasible for CO2 cap {co2_cap_mt:.3f} Mt CO2")
            continue
        
        actual_emissions = calculate_network_emissions(n_scenario)
        cost_increase_pct = (n_scenario.objective - n_baseline.objective) / n_baseline.objective * 100
        
        print(f"  System cost: €{n_scenario.objective:,.0f} ({cost_increase_pct:+.1f}%)")
        print(f"  Actual emissions: {actual_emissions:.3f} Mt CO2 ({actual_emissions/baseline_emissions:.1%} of baseline)")
        
        # Store result
        gen_by_carrier_scenario = get_generation_by_carrier(n_scenario)
        result = {
            "co2_cap_fraction": fraction,
            "co2_cap_mt": co2_cap_mt,
            "system_cost_eur": n_scenario.objective,
            "actual_emissions_mt": actual_emissions,
        }
        
        # Add generator capacities
        for gen in n_scenario.generators.index:
            result[f"capacity_{gen}"] = n_scenario.generators.at[gen, "p_nom_opt"]
        
        # Add battery capacities if present
        if "DK_battery_store" in n_scenario.stores.index:
            result["battery_energy_mwh"] = n_scenario.stores.at["DK_battery_store", "e_nom_opt"]
        if "DK_battery_charger" in n_scenario.links.index:
            result["battery_charger_mw"] = n_scenario.links.at["DK_battery_charger", "p_nom_opt"]
        if "DK_battery_discharger" in n_scenario.links.index:
            result["battery_discharger_mw"] = n_scenario.links.at["DK_battery_discharger", "p_nom_opt"]
        
        # Add generation by carrier
        for carrier, gen_mwh in gen_by_carrier_scenario.items():
            result[f"generation_{carrier}"] = gen_mwh
        
        results.append(result)
        
        # Print dispatch for the strictest CO2 cap scenario
        if i == len(co2_cap_fractions) - 1:
            print(f"\nDispatch for strictest CO2 cap scenario ({fraction:.1%}):")
            
        
            plot_weekly_dispatch(n_scenario, title=f'Weekly Dispatch for {fraction:.1%} CO2 Cap', output_dir=Path("results/co2_sensitivity"))
    
    return pd.DataFrame(results), cost_data, n_baseline


def plot_co2_sensitivity(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create plots showing CO2 sensitivity analysis results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_co2_sensitivity_analysis.
    output_dir : Path
        Directory to save plots.
    """
    # Filter out infinite cap fractions for plotting
    plot_df = results_df[results_df["co2_cap_fraction"] < 1.0].copy()
    plot_df = plot_df.sort_values("co2_cap_mt")
    
    baseline = results_df[results_df["co2_cap_fraction"] == 1.0].iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    
    # --- Plot 1: Generation mix vs CO2 cap ---
    ax = axes[0, 0]
    
    # Find generation columns
    generation_cols = [col for col in plot_df.columns if col.startswith("generation_")]
    
    # Create stacked area chart
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
            values = plot_df[col].values / 1000  # Convert to GWh
            ax.fill_between(x, bottom, bottom + values, label=carrier, alpha=0.7,
                           color=colors.get(carrier, "gray"))
            bottom += values
    
    ax.axhline(y=baseline["generation_solar"] / 1000 + baseline["generation_onwind"] / 1000 + baseline["generation_offwind"] / 1000,
               color="green", linestyle="--", linewidth=1.5, alpha=0.6, label="Baseline renewable")
    ax.set_xlabel("CO2 Cap [Mt CO2/year]")
    ax.set_ylabel("Annual Generation [GWh]")
    ax.set_title("Generation Mix vs CO2 Cap")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # --- Plot 2: System cost vs CO2 cap ---
    ax = axes[0, 1]
    ax.plot(plot_df["co2_cap_mt"], plot_df["system_cost_eur"] / 1e9, marker="o", linewidth=2, markersize=6, color="red")
    ax.axhline(y=baseline["system_cost_eur"] / 1e9, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline: €{baseline['system_cost_eur']/1e9:.2f}B")
    ax.set_xlabel("CO2 Cap [Mt CO2/year]")
    ax.set_ylabel("Annual System Cost [€ Billion]")
    ax.set_title("System Cost vs CO2 Cap")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # --- Plot 3: Actual vs cap emissions ---
    ax = axes[1, 0]
    ax.plot(plot_df["co2_cap_mt"], plot_df["actual_emissions_mt"], marker="o", linewidth=2,
            markersize=6, label="Actual emissions", color="blue")
    ax.plot(plot_df["co2_cap_mt"], plot_df["co2_cap_mt"], linestyle="--", color="red",
            alpha=0.6, label="Cap (limit)")
    ax.axhline(y=baseline["actual_emissions_mt"], color="gray", linestyle=":", alpha=0.5,
               label=f"Baseline: {baseline['actual_emissions_mt']:.3f} Mt CO2")
    ax.set_xlabel("CO2 Cap [Mt CO2/year]")
    ax.set_ylabel("Emissions [Mt CO2/year]")
    ax.set_title("Actual Emissions vs CO2 Cap")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # --- Plot 4: Capacity mix vs CO2 cap ---
    ax = axes[1, 1]
    
    # Plot key generator capacities
    capacity_cols = [col for col in plot_df.columns if col.startswith("capacity_") and any(
        tech in col for tech in ["CCGT", "coal", "solar", "wind", "battery", "nuclear"]
    )]
    
    for col in sorted(capacity_cols):
        gen_name = col.replace("capacity_", "")
        if any(x in gen_name for x in ["CCGT", "coal", "solar", "wind", "battery", "nuclear"]):
            ax.plot(plot_df["co2_cap_mt"], plot_df[col], marker="o", label=gen_name, linewidth=2)
    
    ax.set_xlabel("CO2 Cap [Mt CO2/year]")
    ax.set_ylabel("Installed Capacity [MW]")
    ax.set_title("Key Generator Capacities vs CO2 Cap")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    
    fig.savefig(output_dir / "co2_sensitivity_analysis.png", dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output_dir / 'co2_sensitivity_analysis.png'}")
    plt.close(fig)


def main() -> None:
    """Main execution."""
    
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
        "cost_file": f"cost_data/costs_{financial_parameters['year']}.csv",
        "timeseries_file": "Data/time_series_60min_singleindex_alldata.csv",
    }
    
    # Run sensitivity analysis
    results_df, cost_data, n_baseline = run_co2_sensitivity_analysis(
        cost_file=file_paths["cost_file"],
        timeseries_file=file_paths["timeseries_file"],
        financial_parameters=financial_parameters,
        scenario_parameters=scenario_parameters,
        co2_cap_fractions=[0.8, 0.6, 0.4, 0.2, 0.1],
        co2_price=financial_parameters["co2_price"],
    )
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / "co2_sensitivity_results.csv", index=False)
    print(f"\n✓ Saved results to {OUTPUT_DIR / 'co2_sensitivity_results.csv'}")
    
    # Print full results

    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    summary_cols = ["co2_cap_fraction", "co2_cap_mt", "system_cost_eur", "actual_emissions_mt"]
    print(results_df[summary_cols].to_string(index=False))
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_co2_sensitivity(results_df, OUTPUT_DIR)
    print(f"\n✓ Analysis complete! Results saved to {OUTPUT_DIR}")
    
    # Print generator cost table
    print("\nGenerator Cost Data:")
    gen_techs = ['CCGT', 'coal', 'nuclear', 'solar', 'onwind', 'offwind']
    cost_cols = ['efficiency', 'VOM', 'fuel', 'fixed', 'CO2 intensity']
    available_cols = [col for col in cost_cols if col in cost_data.columns]
    print(cost_data.loc[cost_data.index.isin(gen_techs), available_cols].to_string())
    
    # Print marginal costs with CO2 price
    co2_price = financial_parameters["co2_price"]
    print(f"\nMarginal Costs (with CO2 price €{co2_price}/tCO2):")
    for tech in ['CCGT', 'coal', 'nuclear']:
        if tech in cost_data.index:
            mc = calculate_conventional_marginal_cost(cost_data, tech, co2_price)
            print(f"{tech}: €{mc:.2f}/MWh")
    
    # Print capacity shares
    total_capacity = n_baseline.generators.p_nom_opt.sum()
    print(f"\nCapacity shares (total: {total_capacity:.1f} MW):")
    for gen in n_baseline.generators.index:
        cap = n_baseline.generators.at[gen, 'p_nom_opt']
        share = cap / total_capacity * 100 if total_capacity > 0 else 0
        print(f"{gen}: {cap:.1f} MW ({share:.1f}%)")


if __name__ == "__main__":
    main()
