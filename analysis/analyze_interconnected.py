import pathlib
from turtle import width
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa


def load_network(network_file):
    """Load the PyPSA network from a NetCDF file."""
    return pypsa.Network(network_file)


def create_analysis_folder():
    """Create the results/task_d_analysis folder if it doesn't exist."""
    folder = pathlib.Path("results/task_d_analysis")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def print_system_summary(n):
    """Print a short system summary."""
    print("System Summary:")
    print(f"- Buses: {len(n.buses)}")
    print(f"- Lines: {len(n.lines)}")
    print(f"- Stores: {len(n.stores)}")
    print(f"- Links: {len(n.links)}")

    # Check DK neighbours
    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]
    neighbours = set(dk_lines.bus0).union(set(dk_lines.bus1)) - {"DK"}
    print(f"- DK has {len(neighbours)} neighbours: {sorted(neighbours)}")

    # Check for closed cycle DK-SE-DE-DK
    # Simple check: if lines DK-SE, SE-DE, DE-DK exist
    cycle_lines = [("DK", "SE"), ("SE", "DE"), ("DE", "DK")]
    has_cycle = all(
        any(((line.bus0 == b0 and line.bus1 == b1) or (line.bus0 == b1 and line.bus1 == b0)) for _, line in
            n.lines.iterrows())
        for b0, b1 in cycle_lines
    )
    print(f"- Closed cycle DK-SE-DE-DK: {'Yes' if has_cycle else 'No'}")


def save_line_summary(n, folder):
    """Save line_summary.csv with specified columns."""
    lines = n.lines.copy()
    flows = n.lines_t.p0  # MW, positive direction bus0 to bus1

    utilization = flows.abs().div(lines.s_nom, axis=1)
    max_abs_flow = flows.abs().max()
    mean_abs_flow = flows.abs().mean()
    utilization_max_pu = utilization.max()
    congestion_hours = (utilization >= 0.99).sum()

    summary = pd.DataFrame({
        "line_name": lines.index,
        "bus0": lines.bus0,
        "bus1": lines.bus1,
        "s_nom_MW": lines.s_nom,
        "x": lines.x,
        "max_abs_flow_MW": max_abs_flow,
        "mean_abs_flow_MW": mean_abs_flow,
        "utilization_max_pu": utilization_max_pu,
        "congestion_hours": congestion_hours
    })

    summary.to_csv(folder / "line_summary.csv", index=False)


def save_dk_battery_kpis(n, folder):
    """Save dk_battery_kpis.csv with specified columns."""
    # Check if battery components exist
    has_battery_store = "DK_battery_store" in n.stores.index
    has_charger = "DK_battery_charger" in n.links.index
    has_discharger = "DK_battery_discharger" in n.links.index

    if not (has_battery_store and has_charger and has_discharger):
        print("Battery storage not present in this scenario. Skipping battery KPIs.")
        return

    # Optimized capacities
    energy_capacity = n.stores.loc["DK_battery_store", "e_nom_opt"]
    charge_power = n.links.loc["DK_battery_charger", "p_nom_opt"]
    discharge_power = n.links.loc["DK_battery_discharger", "p_nom_opt"]

    # Annual energy flows
    charger_flow = n.links_t.p0["DK_battery_charger"]  # positive: charging
    discharger_flow = n.links_t.p0["DK_battery_discharger"]  # positive: discharging (from battery to grid)

    annual_charged = charger_flow[charger_flow > 0].sum()  # MWh
    annual_discharged = discharger_flow[discharger_flow > 0].sum()  # MWh

    kpis = pd.DataFrame({
        "optimized_energy_capacity_MWh": [energy_capacity],
        "optimized_charge_power_MW": [charge_power],
        "optimized_discharge_power_MW": [discharge_power],
        "annual_charged_energy_MWh": [annual_charged],
        "annual_discharged_energy_MWh": [annual_discharged]
    })

    kpis.to_csv(folder / "dk_battery_kpis.csv", index=False)


def create_line_loading_summary(n, folder):
    """Create and save line loading summary table."""
    line_labels = {
        "DK_NO": "DK-NO",
        "DK_SE": "DK-SE",
        "DK_DE": "DK-DE",
        "DE_SE": "DE-SE"
    }

    total_hours = len(n.snapshots)
    summary_data = []

    for line_name in n.lines.index:
        flow = n.lines_t.p0[line_name]
        s_nom = n.lines.loc[line_name, "s_nom"]
        loading = flow.abs() / s_nom

        max_loading_pct = round(100 * loading.max(), 1)
        congested_hours = int((loading >= 0.99).sum())
        share_of_year_pct = round(100 * congested_hours / total_hours, 1)

        label = line_labels.get(line_name, line_name)

        summary_data.append({
            "Line": label,
            "Line capacity [MW]": int(round(s_nom)),
            "Congested hours": congested_hours,
            "Share of year [%]": share_of_year_pct,
            "Max loading [%]": max_loading_pct
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values(
        by=["Max loading [%]", "Congested hours"],
        ascending=False
    )

    df.to_csv(folder / "line_loading_summary.csv", index=False)

    print("\nLine Loading Summary:")
    print(df.to_string(index=False))

    latex_str = df.to_latex(index=False, column_format="lcccc", float_format="%.1f")
    with open(folder / "line_loading_summary.tex", "w") as f:
        f.write(latex_str)


def plot_interconnector_summary(n, folder):
    """Create interconnector_summary.png and save data."""
    lines = ["DK_NO", "DK_SE", "DK_DE"]
    data = []

    for line in lines:
        flow = n.lines_t.p0[line]
        s_nom = n.lines.loc[line, "s_nom"]
        bus0 = n.lines.loc[line, "bus0"]

        if bus0 == "DK":
            # positive p0: export from DK
            export_MWh = flow[flow > 0].sum()
            import_MWh = -flow[flow < 0].sum()
        else:
            # bus1 == "DK", positive p0: import to DK
            import_MWh = flow[flow > 0].sum()
            export_MWh = -flow[flow < 0].sum()

        annual_import_GWh = import_MWh / 1000
        annual_export_GWh = export_MWh / 1000
        line_capacity_GW = s_nom / 1000

        loading = flow.abs() / s_nom
        near_cap_hours = (loading >= 0.99).sum()
        total_hours = len(flow)
        share_pct = round(100 * near_cap_hours / total_hours, 1)

        data.append({
            "Line": line.replace("_", "-"),
            "Annual imports to DK [GWh]": annual_import_GWh,
            "Annual exports from DK [GWh]": annual_export_GWh,
            "Line capacity [GW]": line_capacity_GW,
            "Share of year near capacity [%]": share_pct
        })

    df = pd.DataFrame(data)
    df.to_csv(folder / "interconnector_summary.csv", index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(lines))
    width = 0.35

    ax1.bar(x - width / 2, df["Annual imports to DK [GWh]"], width, label='Imports to DK', color='tab:blue', alpha=0.8)
    ax1.bar(x + width / 2, df["Annual exports from DK [GWh]"], width, label='Exports from DK', color='orange',
            alpha=0.8)

    ax1.set_xlabel('Interconnector')
    ax1.set_ylabel('Annual Energy [GWh]')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Line"])
    ax1.grid(True, alpha=0.3)

    # Secondary axis
    ax2 = ax1.twinx()
    line_plot = ax2.plot(x, df["Line capacity [GW]"], 'k-', linewidth=2, markersize=8, label='Line capacity [GW]')
    ax2.set_ylabel('Line Capacity [GW]', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Combined legend
    bars = ax1.get_legend_handles_labels()
    lines_legend = ax2.get_legend_handles_labels()
    ax1.legend(bars[0] + lines_legend[0], bars[1] + lines_legend[1], loc='upper left')

    plt.title("Denmark interconnector use and fixed capacities (2016)")
    plt.tight_layout()
    plt.savefig(folder / "interconnector_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_interconnector_utilisation_comparison(n, folder):
    """Create interconnector utilisation over months chart."""
    # Compute monthly utilisation for each interconnector
    lines = list(n.lines.index)
    monthly_data = {}

    for line in lines:
        flow = n.lines_t.p0[line]
        s_nom = n.lines.loc[line, "s_nom"]
        loading = flow.abs() / s_nom * 100  # %
        monthly_avg = loading.groupby(flow.index.month).mean()
        monthly_data[line.replace("_", "-")] = monthly_avg

    df = pd.DataFrame(monthly_data)
    df.index.name = "Month"
    df.to_csv(folder / "interconnector_utilisation_comparison.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(1, 13)

    for interconnector in df.columns:
        ax.plot(x, df[interconnector], marker='o', linewidth=2, label=interconnector)

    ax.set_xlabel("Month")
    ax.set_ylabel("Utilisation [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_title("Interconnector utilisation over months (2016)")

    plt.tight_layout()
    plt.savefig(folder / "interconnector_utilisation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_denmark_export_origin(n, folder):
    """Analyze the origin of Denmark's exports: local surplus vs re-exported imports."""
    # DK generation: sum of all generator outputs at bus DK
    dk_gens = n.generators[n.generators.bus == "DK"]
    dk_generation = n.generators_t.p[dk_gens.index].sum(axis=1)

    # DK load
    dk_load = n.loads_t.p_set["DK_electricity_demand"]

    # Battery: charging (positive consumption), discharging (positive generation)
    battery_charge = pd.Series(0, index=n.snapshots)
    battery_discharge = pd.Series(0, index=n.snapshots)
    if "DK_battery_charger" in n.links.index:
        battery_charge = n.links_t.p0["DK_battery_charger"]  # positive: charging
    if "DK_battery_discharger" in n.links.index:
        battery_discharge = n.links_t.p0["DK_battery_discharger"]  # positive: discharging

    # DK imports and exports - correct sign-based calculation
    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]
    imports = pd.Series(0, index=n.snapshots)
    exports = pd.Series(0, index=n.snapshots)

    for line in dk_lines.index:
        flow = n.lines_t.p0[line]
        if n.lines.loc[line, "bus0"] == "DK":
            # positive p0: export from DK
            exports += flow.clip(lower=0)
            # negative p0: import to DK
            imports += (-flow).clip(lower=0)
        else:  # bus1 == "DK"
            # positive p0: import to DK
            imports += flow.clip(lower=0)
            # negative p0: export from DK
            exports += (-flow).clip(lower=0)

    # Calculate per hour
    reexport = pd.Series([min(e, i) for e, i in zip(exports, imports)], index=n.snapshots)
    local_export = pd.Series([max(e - i, 0) for e, i in zip(exports, imports)], index=n.snapshots)

    # Aggregate by month
    monthly_data = []
    for month in range(1, 13):
        mask = n.snapshots.month == month
        total_exports = exports[mask].sum() / 1000  # GWh
        local_exp = local_export[mask].sum() / 1000
        reexp = reexport[mask].sum() / 1000
        total_imports = imports[mask].sum() / 1000

        monthly_data.append({
            "Month": month,
            "Exports total [GWh]": total_exports,
            "Exports from DK surplus [GWh]": local_exp,
            "Re-exported imports [GWh]": reexp,
            "Imports [GWh]": total_imports
        })

    df = pd.DataFrame(monthly_data)

    # Sanity check
    df["Check"] = df["Exports total [GWh]"] - df["Exports from DK surplus [GWh]"] - df["Re-exported imports [GWh]"]
    print("\nDenmark Export Origin Monthly Summary:")
    print(df.to_string(index=False))

    # Remove check column before saving
    df = df.drop(columns=["Check"])
    df.to_csv(folder / "denmark_export_origin.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(1, 13)

    # Stacked bars
    ax.bar(x, df["Exports from DK surplus [GWh]"], label='Exports from DK surplus', color='orange', alpha=0.8)
    ax.bar(x, df["Re-exported imports [GWh]"], bottom=df["Exports from DK surplus [GWh]"], label='Re-exported imports',
           color='navajowhite', alpha=0.9)

    # Black line for total exports
    ax.plot(x, df["Exports total [GWh]"], 'k-', linewidth=2, label='Total exports')

    ax.set_xlabel("Month")
    ax.set_ylabel("Exports [GWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()
    ax.set_title("Origin of Denmark exports in the 2016 model case")

    plt.tight_layout()
    plt.savefig(folder / "denmark_export_origin.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_data_for_last_step(n, folder):
    """Save CSV with first-timestep bus imbalances and line flows."""
    first_ts = n.snapshots[0]

    rows = []

    # ---- Bus data: generation, demand, imbalance ----
    for bus in n.buses.index:
        gens_at_bus = n.generators[n.generators.bus == bus]
        gen_total = n.generators_t.p.loc[first_ts, gens_at_bus.index].sum()

        loads_at_bus = n.loads[n.loads.bus == bus]
        demand_total = n.loads_t.p_set.loc[first_ts, loads_at_bus.index].sum()

        imbalance = gen_total - demand_total

        rows.append({
            "Type": "bus",
            "Name": bus,
            "bus0": "",
            "bus1": "",
            "Generation [MW]": gen_total,
            "Demand [MW]": demand_total,
            "Imbalance [MW]": imbalance,
            "Flow p0 [MW]": np.nan
        })

    # ---- Line data: first-timestep power flows ----
    for line in n.lines.index:
        flow = n.lines_t.p0.loc[first_ts, line]

        rows.append({
            "Type": "line",
            "Name": line,
            "bus0": n.lines.loc[line, "bus0"],
            "bus1": n.lines.loc[line, "bus1"],
            "Generation [MW]": np.nan,
            "Demand [MW]": np.nan,
            "Imbalance [MW]": np.nan,
            "Flow p0 [MW]": flow
        })

    df = pd.DataFrame(rows)
    df.to_csv(folder / "data_for_last_step.csv", index=False)


def save_generation_summary_by_country(n, folder):
    """Save and plot optimized generation capacity for each country, broken down by technology."""

    rows = []

    for gen in n.generators.index:
        bus = n.generators.loc[gen, "bus"]
        carrier = n.generators.loc[gen, "carrier"]

        # Optimized capacity if available, otherwise fixed capacity
        if "p_nom_opt" in n.generators.columns:
            capacity = n.generators.loc[gen, "p_nom_opt"]
        else:
            capacity = n.generators.loc[gen, "p_nom"]

        # Annual generation
        annual_generation_MWh = n.generators_t.p[gen].sum()
        annual_generation_GWh = annual_generation_MWh / 1000

        rows.append({
            "Country": bus,
            "Generator": gen,
            "Carrier": carrier,
            "Optimized capacity [MW]": capacity,
            "Annual generation [GWh]": annual_generation_GWh
        })

    df = pd.DataFrame(rows)

    # Save full generator-level data
    df.to_csv(folder / "generation_by_generator_and_country.csv", index=False)

    # Aggregate by country and carrier
    df_by_country_carrier = df.groupby(["Country", "Carrier"], as_index=False).agg({
        "Optimized capacity [MW]": "sum",
        "Annual generation [GWh]": "sum"
    })

    df_by_country_carrier.to_csv(folder / "generation_by_country_and_carrier.csv", index=False)

    # Print nice summary
    print("\nGeneration Summary by Country and Carrier:")
    print(df_by_country_carrier.to_string(index=False))

    # ---------- Plot: optimized capacity by country, split by technology ----------
    plot_df = df_by_country_carrier.pivot(
        index="Country",
        columns="Carrier",
        values="Optimized capacity [MW]"
    ).fillna(0)

    # Optional: keep countries in a nice order if they exist
    preferred_order = [c for c in ["DK", "DE", "SE", "NO"] if c in plot_df.index]
    remaining = [c for c in plot_df.index if c not in preferred_order]
    plot_df = plot_df.loc[preferred_order + remaining]

    # Optional: keep technologies in a consistent order if they exist
    tech_order = [t for t in ["coal", "gas", "nuclear", "offwind", "onwind", "solar"] if t in plot_df.columns]
    remaining_tech = [t for t in plot_df.columns if t not in tech_order]
    plot_df = plot_df[tech_order + remaining_tech]

    # Color palette
    color_map = {
        "coal": "#4C566A",  # muted slate
        "gas": "#D08770",  # requested
        "nuclear": "#7AA66D",  # muted green
        "offwind": "#2E86AB",  # requested
        "onwind": "#8F6FB5",  # soft purple
        "solar": "#EBCB3B"  # requested
    }

    colors = [color_map.get(col, "#999999") for col in plot_df.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", stacked=True, ax=ax, width=0.75, color=colors)

    ax.set_xlabel("Country")
    ax.set_ylabel("Optimized capacity [MW]")
    ax.set_title("Optimized generation capacity by country and technology")
    ax.set_ylim(0, 150000)
    ax.legend(title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(folder / "generation_capacity_by_country_and_technology.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_denmark_dispatch_strategy(n, folder):
    """
    Plotd Denmark's dispatch strategy for the winter week with the highest
    average Danish electricity demand.

    The figure shows:
    - Danish domestic generation by carrier (stacked area)
    - imports from each neighbouring country (positive dashed lines)
    - exports to each neighbouring country (negative dashed lines)
    - Danish demand (black line)

    Saves:
        denmark_dispatch_strategy_winter_week.png
    """

    snapshots = n.snapshots

    # Winter = December + January
    winter_mask = (snapshots.month == 12) | (snapshots.month == 1)

    # Denmark load
    dk_load = n.loads_t.p_set["DK_electricity_demand"]
    winter_load = dk_load[winter_mask]

    # Find winter week with highest average load
    weekly_avg_load = winter_load.resample("W").mean()
    max_load_week_end = weekly_avg_load.idxmax()
    week_start = max_load_week_end - pd.Timedelta(days=6)
    week_end = max_load_week_end

    week_index = dk_load.loc[week_start:week_end].index
    week_load = dk_load.loc[week_index]

    # -----------------------------
    # 1. Denmark domestic generation by carrier
    # -----------------------------
    dk_generators = n.generators[n.generators.bus == "DK"]

    generation_by_carrier = {}
    for gen in dk_generators.index:
        carrier = n.generators.at[gen, "carrier"]

        if carrier not in generation_by_carrier:
            generation_by_carrier[carrier] = pd.Series(0.0, index=week_index)

        generation_by_carrier[carrier] = (
            generation_by_carrier[carrier]
            .add(n.generators_t.p[gen].loc[week_index], fill_value=0.0)
        )

    # Optional: include storage discharge/charge if your model uses links
    # Battery charging is demand-like, battery discharging is supply-like
    if "DK_battery_discharger" in n.links.index:
        generation_by_carrier["battery_discharge"] = n.links_t.p1["DK_battery_discharger"].loc[week_index].clip(lower=0)

    battery_charge = pd.Series(0.0, index=week_index)
    if "DK_battery_charger" in n.links.index:
        battery_charge = n.links_t.p0["DK_battery_charger"].loc[week_index].clip(lower=0)
    # charging consumes power from DK bus
        # depending on model convention p0 may already be positive when consuming

    # -----------------------------
    # 2. Imports/exports by neighbour
    # -----------------------------
    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]

    exchanges = {}

    for line in dk_lines.index:
        bus0 = n.lines.at[line, "bus0"]
        bus1 = n.lines.at[line, "bus1"]

        neighbour = bus1 if bus0 == "DK" else bus0
        flow = n.lines_t.p0[line].loc[week_index]

        # PyPSA convention:
        # p0 > 0 means power flows from bus0 -> bus1
        # Convert to "positive = import to DK, negative = export from DK"
        if bus0 == "DK":
            dk_exchange = -flow
        else:
            dk_exchange = flow

        if neighbour not in exchanges:
            exchanges[neighbour] = pd.Series(0.0, index=week_index)

        exchanges[neighbour] = exchanges[neighbour].add(dk_exchange, fill_value=0.0)

    # -----------------------------
    # 3. Plot (two panels)
    # -----------------------------

    preferred_order = [
    "solar", "onwind", "offwind",
    "gas", "coal", "nuclear",
    "battery_discharge"
    ]

    remaining = [c for c in generation_by_carrier if c not in preferred_order]

    carrier_order = [c for c in preferred_order if c in generation_by_carrier] + remaining

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # -----------------------------
    # TOP PANEL: DK generation
    # -----------------------------
    colors = {
        "solar": "#ffd92f",
        "onwind": "#1b9e77",
        "offwind": "#377eb8",
        "gas": "#e41a1c",
        "coal": "#4d4d4d",
        "nuclear": "#984ea3",
        "battery_discharge": "#ff7f00",
    }

    bottom = np.zeros(len(week_index))

    for carrier in carrier_order:
        series = generation_by_carrier[carrier].fillna(0).values

        ax1.fill_between(
            week_index,
            bottom,
            bottom + series,
            label=f"DK {carrier}",
            color=colors.get(carrier, None),
            alpha=0.8
        )

        bottom += series

    # Demand including battery charging
    effective_demand = week_load + battery_charge
    ax1.plot(
        week_index,
        effective_demand,
        color="black",
        linewidth=2.3,
        label="DK demand"
    )

    ax1.set_ylabel("Power [MW]")
    ax1.set_title("Denmark dispatch strategy during peak winter week")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax1.grid(True, alpha=0.3)

    # -----------------------------
    # BOTTOM PANEL: Exchanges
    # -----------------------------
    exchange_colors = {
        "DE": "#2ca02c",
        "SE": "#ff7f0e",
        "NO": "#1f77b4"
    }

    for neighbour, series in exchanges.items():
        ax2.plot(
            week_index,
            series,
            linewidth=2.5,
            label=f"{neighbour}",
            color=exchange_colors.get(neighbour, None)
        )

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Import / Export [MW]")
    ax2.set_xlabel("Time")
    ax2.legend(title="Exchange", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    outfile = folder / "denmark_dispatch_strategy_winter_week.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function to run the analysis."""
    # Scenario metadata (not used for loading, but for context)
    weather_year = 2016
    countries = ["DK", "DE", "SE", "NO"]
    with_battery_storage = True
    with_interconnectors = True

    network_file = "results\\networks\\interconnected_2016.nc"
    n = load_network(network_file)

    folder = create_analysis_folder()

    print_system_summary(n)

    save_line_summary(n, folder)
    save_dk_battery_kpis(n, folder)
    create_line_loading_summary(n, folder)

    plot_interconnector_summary(n, folder)
    plot_interconnector_utilisation_comparison(n, folder)
    analyze_denmark_export_origin(n, folder)
    save_data_for_last_step(n, folder)
    save_generation_summary_by_country(n, folder)
    plot_denmark_dispatch_strategy(n, folder)

    print(f"Analysis complete. Files saved in {folder}")


if __name__ == "__main__":
    main()