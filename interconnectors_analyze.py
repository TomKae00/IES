import pathlib
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
        any(((line.bus0 == b0 and line.bus1 == b1) or (line.bus0 == b1 and line.bus1 == b0)) for _, line in n.lines.iterrows())
        for b0, b1 in cycle_lines
    )
    print(f"- Closed cycle DK-SE-DE-DK: {'Yes' if has_cycle else 'No'}")


def save_line_summary(n, folder):
    """Save line_summary.csv with specified columns."""
    lines = n.lines.copy()
    flows = n.lines_t.p0  # MW, positive direction bus0 to bus1

    # Compute metrics
    max_abs_flow = flows.abs().max()
    mean_abs_flow = flows.abs().mean()
    utilization_max_pu = max_abs_flow / lines.s_nom
    congestion_hours = (flows.abs() > lines.s_nom).sum()

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
    # Map line names to readable labels
    line_labels = {
        "DK_NO": "DK-NO",
        "DK_SE": "DK-SE",
        "DK_DE": "DK-DE",
        "DE_SE": "DE-SE"
    }

    summary_data = []
    for line_name in n.lines.index:
        flow = n.lines_t.p0[line_name]
        s_nom = n.lines.loc[line_name, "s_nom"]
        loading = flow.abs() / s_nom
        max_loading_pct = round(100 * loading.max(), 1)
        congested_hours = (loading >= 0.99).sum()
        label = line_labels.get(line_name, line_name)
        summary_data.append({
            "Line": label,
            "Congested hours": congested_hours,
            "Max loading [%]": max_loading_pct
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values("Max loading [%]", ascending=False)

    # Save CSV
    df.to_csv(folder / "line_loading_summary.csv", index=False)

    # Print to terminal
    print("\nLine Loading Summary:")
    print(df.to_string(index=False))

    # Save LaTeX
    latex_str = df.to_latex(index=False, column_format="lcc", float_format="%.1f")
    with open(folder / "line_loading_summary.tex", "w") as f:
        f.write(latex_str)


def plot_topology(n, folder):
    """Create topology_task_d.png with manual coordinates."""
    coords = {
        "DK": (0, 0),
        "DE": (1.2, -0.8),
        "SE": (1.2, 0.8),
        "NO": (0.2, 1.6)
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot buses
    for bus, (x, y) in coords.items():
        ax.plot(x, y, 'ko', markersize=10)
        ax.text(x, y + 0.1, bus, ha='center', va='bottom', fontsize=12)

    # Plot lines
    for _, line in n.lines.iterrows():
        b0, b1 = line.bus0, line.bus1
        if b0 in coords and b1 in coords:
            x0, y0 = coords[b0]
            x1, y1 = coords[b1]
            ax.plot([x0, x1], [y0, y1], 'b-', linewidth=2)
            # Label capacity at midpoint
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mid_x, mid_y, f"{line.s_nom:.0f} MW", ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(-1.2, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(folder / "topology_task_d.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_line_loading_duration_curves(n, folder):
    """Create line_loading_duration_curves.png."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for line_name in n.lines.index:
        flow = n.lines_t.p0[line_name]
        s_nom = n.lines.loc[line_name, "s_nom"]
        loading = flow.abs() / s_nom
        sorted_loading = np.sort(loading.values)[::-1]  # descending
        hours = np.arange(1, len(sorted_loading) + 1)
        ax.plot(hours, sorted_loading, label=line_name)

    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("Sorted hours")
    ax.set_ylabel("Loading [-]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "line_loading_duration_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_denmark_dispatch(n, folder):
    """Create denmark_dispatch_imports_battery_winter_week.png."""
    # Find winter weeks: December and January
    snapshots = n.snapshots
    winter_mask = (snapshots.month == 12) | (snapshots.month == 1)

    # Denmark load
    dk_load = n.loads_t.p_set["DK_electricity_demand"]
    winter_load = dk_load[winter_mask]

    # Group by week and find week with highest average load
    weekly_avg_load = winter_load.groupby(pd.Grouper(freq='W')).mean()
    max_load_week = weekly_avg_load.idxmax()
    week_start = max_load_week - pd.Timedelta(days=6)  # assuming week starts on Monday or adjust
    week_end = max_load_week

    # Actually, to get the week, better to resample
    week_data = dk_load.loc[week_start:week_end]

    # Generation by carrier
    dk_generators = n.generators[n.generators.bus == "DK"]
    generation = {}
    carriers = []
    for gen in dk_generators.index:
        carrier = n.generators.loc[gen, "carrier"]
        if carrier not in generation:
            generation[carrier] = pd.Series(0, index=week_data.index)
            carriers.append(carrier)
        generation[carrier] += n.generators_t.p[gen].loc[week_data.index]

    # Net imports: sum of line flows into DK (positive if importing)
    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]
    imports = pd.Series(0, index=week_data.index)
    for line in dk_lines.index:
        flow = n.lines_t.p0[line].loc[week_data.index]
        if n.lines.loc[line, "bus0"] == "DK":
            imports -= flow  # if flow positive, exporting
        else:
            imports += flow  # if flow positive from other to DK, importing

    # Battery: charging negative (consuming), discharging positive
    battery_charge = -n.links_t.p0["DK_battery_charger"].loc[week_data.index]  # negative for charging
    battery_discharge = n.links_t.p0["DK_battery_discharger"].loc[week_data.index]  # positive for discharging

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Stacked generation
    bottom = np.zeros(len(week_data))
    colors = {'solar': '#ffd92f', 'onwind': '#1b9e77', 'offwind': '#377eb8', 'gas': '#e41a1c', 'coal': '#4d4d4d', 'nuclear': '#984ea3'}
    for carrier in carriers:
        if carrier in colors:
            ax.fill_between(week_data.index, bottom, bottom + generation[carrier], label=carrier, color=colors[carrier], alpha=0.7)
            bottom += generation[carrier]

    # Net imports
    ax.plot(week_data.index, imports, 'k-', linewidth=2, label='Net imports')

    # Battery
    ax.plot(week_data.index, battery_charge, 'r--', linewidth=1, label='Battery charging')
    ax.plot(week_data.index, battery_discharge, 'b--', linewidth=1, label='Battery discharging')

    ax.set_xlabel("Time")
    ax.set_ylabel("Power [MW]")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(folder / "denmark_dispatch_imports_battery_winter_week.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_denmark_dispatch_strategy(n, folder):
    """
    Plot Denmark's dispatch strategy for the winter week with the highest
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
        # charging consumes power from DK bus
        # depending on model convention p0 may already be positive when consuming
        battery_charge = n.links_t.p0["DK_battery_charger"].loc[week_index].clip(lower=0)

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

    # Demand
    ax1.plot(
        week_index,
        week_load,
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

    network_file = "results/regional_network_2016.nc"
    n = load_network(network_file)

    folder = create_analysis_folder()

    print_system_summary(n)

    save_line_summary(n, folder)
    save_dk_battery_kpis(n, folder)
    create_line_loading_summary(n, folder)

    plot_topology(n, folder)
    plot_line_loading_duration_curves(n, folder)
    plot_denmark_dispatch(n, folder)
    plot_denmark_dispatch_strategy(n, folder)

    print(f"Analysis complete. Files saved in {folder}")


if __name__ == "__main__":
    main()