from pathlib import Path
import argparse

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import model

ELECTRICITY_TOPOLOGY = model.PIPELINE_TOPOLOGY

OUTPUT_DIR = Path("results/gas_pipelines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SCENARIO = {
    "weather_year": "2016",
    "countries": ["DK", "DE", "SE", "NO"],
    "with_battery_storage": True,
    "with_interconnectors": True,
    "co2_price": 80.0,
}

FINANCIAL_PARAMETERS = {
    "fill_values": 0.0,
    "r": 0.07,
    "nyears": 1,
    "year": 2025,
}

FILE_PATHS = {
    "cost_file": f"cost_data/costs_{FINANCIAL_PARAMETERS['year']}.csv",
    "timeseries_file": "Data/time_series_60min_singleindex_alldata.csv",
}

DEFAULT_DEBUG_SNAPSHOT_COUNT = 48


def build_base_network(all_timeseries_data: dict) -> pypsa.Network:
    cost_data = model.prepare_costs(
        cost_file=FILE_PATHS["cost_file"],
        financial_parameters=FINANCIAL_PARAMETERS,
        number_of_years=FINANCIAL_PARAMETERS["nyears"],
    )

    n = model.create_regional_network(
        cost_data=cost_data,
        all_timeseries_data=all_timeseries_data,
        co2_price=DEFAULT_SCENARIO["co2_price"],
        with_battery_storage=DEFAULT_SCENARIO["with_battery_storage"],
        with_interconnectors=DEFAULT_SCENARIO["with_interconnectors"],
    )
    return n


def optimize_network(n: pypsa.Network, output_file: Path) -> pypsa.Network:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    n.optimize(
        n.snapshots,
        extra_functionality=model.custom_constraints,
        solver_name="gurobi",
        include_objective_constant=True,
    )
    n.export_to_netcdf(output_file)
    print(f"Optimized network saved to {output_file}")
    return n
    output_file.parent.mkdir(parents=True, exist_ok=True)
    n.optimize(
        n.snapshots,
        extra_functionality=model.custom_constraints,
        solver_name="gurobi",
        include_objective_constant=True,
    )
    n.export_to_netcdf(output_file)
    print(f"Optimized network saved to {output_file}")
    return n


def compute_link_energy(n: pypsa.Network, link_names: list[str]) -> float:
    energy = 0.0
    for link_name in link_names:
        if link_name not in n.links.index:
            continue
        if "p0" not in n.links_t or link_name not in n.links_t.p0.columns:
            continue
        energy += float(n.links_t.p0[link_name].abs().sum())
    return energy


def compute_route_energies(n: pypsa.Network, link_names: list[str]) -> dict[str, float]:
    route_energies = {}
    for link_name in link_names:
        if link_name not in n.links.index:
            route_energies[link_name] = 0.0
            continue
        if "p0" not in n.links_t or link_name not in n.links_t.p0.columns:
            route_energies[link_name] = 0.0
            continue
        route_energies[link_name] = float(n.links_t.p0[link_name].abs().sum())
    return route_energies


def compute_line_energy(n: pypsa.Network, line_name: str) -> float:
    if line_name not in n.lines.index:
        return 0.0
    if "p0" not in n.lines_t or line_name not in n.lines_t.p0.columns:
        return 0.0
    return float(n.lines_t.p0[line_name].abs().sum())


def compute_lines_energy(n: pypsa.Network, line_names: list[str]) -> float:
    energy = 0.0
    for line_name in line_names:
        energy += compute_line_energy(n, line_name)
    return energy


def get_interconnector_line_names() -> list[str]:
    return [f"{bus0}_{bus1}" for bus0, bus1 in ELECTRICITY_TOPOLOGY]


def load_all_country_timeseries(snapshot_count: int | None = None) -> dict[str, dict[str, pd.Series]]:
    all_timeseries_data = {}
    for country_code in DEFAULT_SCENARIO["countries"]:
        data = model.load_country_timeseries(
            timeseries_file=FILE_PATHS["timeseries_file"],
            country_code=country_code,
            year=DEFAULT_SCENARIO["weather_year"],
        )
        if snapshot_count is not None:
            data = {key: series.iloc[:snapshot_count] for key, series in data.items()}
        all_timeseries_data[country_code] = data
    return all_timeseries_data


def load_or_optimize(n: pypsa.Network, output_file: Path, force: bool) -> pypsa.Network:
    if output_file.exists() and not force:
        print(f"Output already exists, loading from {output_file}")
        return pypsa.Network(output_file)

    print(f"Optimizing and saving network to {output_file}")
    return optimize_network(n, output_file)


def get_pipeline_flow_by_link(n: pypsa.Network, pipeline_type: str) -> dict[str, float]:
    prefix = f"{pipeline_type}_pipeline_"
    flows = {}  
    if "p0" not in n.links_t:
        return flows

    for link_name in n.links.index:
        if not link_name.startswith(prefix):
            continue
        if link_name not in n.links_t.p0.columns:
            continue
        flows[link_name] = float(n.links_t.p0[link_name].abs().sum()) / 1e3
    return flows
    prefix = f"{pipeline_type}_pipeline_"
    flows = {}  
    if "p0" not in n.links_t:
        return flows

    for link_name in n.links.index:
        if not link_name.startswith(prefix):
            continue
        if link_name not in n.links_t.p0.columns:
            continue
        flows[link_name] = float(n.links_t.p0[link_name].abs().sum()) / 1e3
    return flows


def plot_gas_pipeline_flow_networks(output_dir: Path, pipeline_types: list[str] = None) -> Path | None:
    if pipeline_types is None:
        pipeline_types = list(model.PIPELINE_REFERENCE.keys())

    positions = {
        "DK": (0.35, 0.45),
        "DE": (0.35, 0.15),
        "NO": (0.15, 0.80),
        "SE": (0.75, 0.45),
    }

    flow_data = {}
    max_flow = 0.0
    for pipeline_type in pipeline_types:
        path = output_dir / f"{pipeline_type}_pipeline_{DEFAULT_SCENARIO['weather_year']}.nc"
        if not path.exists():
            print(f"Skipping {pipeline_type}: output file not found at {path}")
            continue
        n = pypsa.Network(path)
        flow_data[pipeline_type] = get_pipeline_flow_by_link(n, pipeline_type)
        max_flow = max(max_flow, max(flow_data[pipeline_type].values(), default=0.0))

    if not flow_data:
        return None

    norm = Normalize(vmin=0.0, vmax=max_flow if max_flow > 0 else 1.0)
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, len(flow_data), figsize=(12, 6), constrained_layout=True)
    if len(flow_data) == 1:
        axes = [axes]

    for ax, (pipeline_type, flows) in zip(axes, flow_data.items()):
        ax.set_title(f"{pipeline_type} pipeline flows")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")

        #for country, (x, y) in positions.items():
         #   ax.text(x, y, country, fontsize=12, fontweight="bold", ha="center", va="center",
          #          bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))

        for bus0, bus1 in model.PIPELINE_TOPOLOGY:
            link_name = f"{pipeline_type}_pipeline_{bus0}_{bus1}"
            value = flows.get(link_name, 0.0)
            start = positions[bus0]
            end = positions[bus1]
            color = cmap(norm(value))
            linewidth = 1.0 + 4.0 * (value / max_flow if max_flow > 0 else 0)
            arrowprops = dict(
                arrowstyle='-|>',
                color=color,
                linewidth=linewidth,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=18,
            )
            ax.annotate("", xy=end, xytext=start, arrowprops=arrowprops)
            if value > 0.0:
                mid_x = 0.5 * (start[0] + end[0])
                mid_y = 0.5 * (start[1] + end[1])
                ax.text(mid_x, mid_y, f"{value:.1f}", fontsize=8, ha="center", va="center", color="black",
                        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.2"))

        for country, (x, y) in positions.items():
            ax.text(x, y, country, fontsize=12, fontweight="bold", ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
            
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Annual flow (GWh)")
    fig.suptitle("Average gas pipeline flows — CH₄ vs H₂", fontsize=14, y=1.02)

    plot_file = output_dir / "pipeline_flow_network.png"
    fig.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_file


def normalize_carrier_name(carrier: str) -> str:
    if carrier is None:
        return ""
    text = str(carrier).strip().lower()
    text = text.replace("₄", "4").replace("₂", "2")
    text = text.replace(" ", "").replace("_", "")
    return text


def get_pipeline_route_transport(n: pypsa.Network, carrier: str) -> dict[str, float]:
    route_flows = {}
    if "p0" not in n.links_t:
        return route_flows

    carrier_key = normalize_carrier_name(carrier)
    print("DEBUG n.links carriers and routes:")
    print(n.links[["carrier", "bus0", "bus1"]].to_string())

    matching_links = []
    for link_name, link_row in n.links.iterrows():
        if normalize_carrier_name(link_row["carrier"]) != carrier_key:
            continue
        if link_name not in n.links_t.p0.columns:
            continue
        matching_links.append(link_name)

    if not matching_links:
        return route_flows

    snapshot_weight = 1.0
    if hasattr(n, "snapshot_weightings") and hasattr(n.snapshot_weightings, "generators"):
        try:
            snapshot_weight = float(n.snapshot_weightings.generators.mean())
        except Exception:
            snapshot_weight = 1.0

    flows = n.links_t.p0[matching_links].clip(lower=0).sum()
    for link_name in matching_links:
        if link_name not in flows.index:
            continue
        bus0 = n.links.loc[link_name, "bus0"]
        bus1 = n.links.loc[link_name, "bus1"]
        route_label = f"{bus0}-{bus1}"
        route_flows[route_label] = float(flows[link_name] * snapshot_weight / 1000.0)
    return route_flows


def plot_annual_pipeline_route_transport_by_carrier(
    output_dir: Path,
    n: pypsa.Network | None = None,
    n_ch4: pypsa.Network | None = None,
    n_h2: pypsa.Network | None = None,
) -> Path | None:
    if n is None and n_ch4 is None and n_h2 is None:
        print("No network provided for route transport plotting.")
        return None

    ch4_flows = {}
    h2_flows = {}
    if n_ch4 is not None:
        ch4_flows = get_pipeline_route_transport(n_ch4, "CH4")
    if n_h2 is not None:
        h2_flows = get_pipeline_route_transport(n_h2, "H2")

    # If separate CH4/H2 networks were not provided, try to extract both from the single network.
    if n is not None and (not ch4_flows or not h2_flows):
        if not ch4_flows:
            ch4_flows = get_pipeline_route_transport(n, "CH4")
        if not h2_flows:
            h2_flows = get_pipeline_route_transport(n, "H2")

    print("DEBUG route transport CH4:", ch4_flows)
    print("DEBUG route transport H2:", h2_flows)

    routes = sorted(set(ch4_flows) | set(h2_flows))
    if not routes:
        print("No pipeline routes found for CH4 or H2.")
        return None

    ch4_values = [ch4_flows.get(route, 0.0) for route in routes]
    h2_values = [h2_flows.get(route, 0.0) for route in routes]

    y_positions = list(range(len(routes)))
    bar_height = 0.35
    ch4_positions = [y - bar_height / 2 for y in y_positions]
    h2_positions = [y + bar_height / 2 for y in y_positions]

    colors = {
        "CH4": "#d97904",
        "H2": "#2a7f9e",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ch4_positions, ch4_values, height=bar_height, color=colors["CH4"], label="CH₄")
    ax.barh(h2_positions, h2_values, height=bar_height, color=colors["H2"], label="H₂")

    max_value = max(ch4_values + h2_values) if ch4_values or h2_values else 1.0
    for y, value in zip(ch4_positions, ch4_values):
        ax.text(value + max_value * 0.01, y, f"{value:.1f}", va='center', fontsize=9)
    for y, value in zip(h2_positions, h2_values):
        ax.text(value + max_value * 0.01, y, f"{value:.1f}", va='center', fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(routes)
    ax.set_xlabel("Annual transport (GWh)")
    ax.set_title("Annual gas pipeline transport by route and carrier")
    ax.legend()
    fig.tight_layout()

    plot_file = output_dir / "annual_gas_pipeline_transport_by_route_and_carrier.png"
    fig.savefig(plot_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return plot_file


def load_saved_pipeline_network(pipeline_type: str) -> pypsa.Network | None:
    path = OUTPUT_DIR / f"{pipeline_type}_pipeline_{DEFAULT_SCENARIO['weather_year']}.nc"
    if not path.exists():
        print(f"Saved gas pipeline network not found: {path}")
        return None
    return pypsa.Network(path)
    path = OUTPUT_DIR / f"{pipeline_type}_pipeline_{DEFAULT_SCENARIO['weather_year']}.nc"
    if not path.exists():
        print(f"Saved gas pipeline network not found: {path}")
        return None
    return pypsa.Network(path)


def run_baseline(base_network: pypsa.Network, force: bool) -> tuple[pypsa.Network, float]:
    baseline_n = base_network.copy()
    baseline_out = OUTPUT_DIR / f"baseline_electricity_{DEFAULT_SCENARIO['weather_year']}.nc"
    baseline_n = load_or_optimize(baseline_n, baseline_out, force=force)
    baseline_energy_gwh = compute_lines_energy(baseline_n, get_interconnector_line_names()) / 1e3
    print(
        f"\nBaseline electricity energy on the full interconnector network: "
        f"{baseline_energy_gwh:.1f} GWh"
    )
    return baseline_n, baseline_energy_gwh


def run_scenario(pipeline_type: str, base_network: pypsa.Network, force: bool) -> dict:
    scenario_name = f"{pipeline_type}_pipeline"
    n = base_network.copy()
    link_names = model.add_gas_pipeline(n, pipeline_type)
    output_file = OUTPUT_DIR / f"{scenario_name}_{DEFAULT_SCENARIO['weather_year']}.nc"
    n = load_or_optimize(n, output_file, force=force)

    pipeline_energy_mwh = compute_link_energy(n, link_names)
    route_energies_mwh = compute_route_energies(n, link_names)
    electricity_interconnector_energy_mwh = compute_lines_energy(n, get_interconnector_line_names())

    result = {
        "scenario": scenario_name,
        "objective_eur": float(n.objective),
        "pipeline_energy_gwh": pipeline_energy_mwh / 1e3,
        "electricity_interconnector_energy_gwh": electricity_interconnector_energy_mwh / 1e3,
        "pipeline_capacity_mw": model.PIPELINE_REFERENCE[pipeline_type]["p_nom_mw"] * len(link_names),
        "pipeline_routes": len(link_names),
        "pipeline_links": link_names,
    }
    for link_name, energy_mwh in route_energies_mwh.items():
        route_label = link_name.split(f"{pipeline_type}_pipeline_")[-1]
        result[f"route_{route_label}_gwh"] = energy_mwh / 1e3

    print(f"\nScenario: {scenario_name}")
    print(f"  Objective: {result['objective_eur']:.0f} EUR")
    print(f"  Pipeline capacity (total across {result['pipeline_routes']} routes): {result['pipeline_capacity_mw']} MW")
    print(f"  Pipeline energy transported: {result['pipeline_energy_gwh']:.1f} GWh")
    print(
        f"  Electricity interconnector energy transported: "
        f"{result['electricity_interconnector_energy_gwh']:.1f} GWh"
    )
    print(f"  Pipeline routes: {', '.join(link_names)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gas pipeline comparison scenarios using the existing regional electricity network model."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(model.PIPELINE_REFERENCE.keys()),
        default=list(model.PIPELINE_REFERENCE.keys()),
        help="Pipeline scenarios to run. Default: all.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a reduced snapshot debug version of the model.",
    )
    parser.add_argument(
        "--snapshot-count",
        type=int,
        default=None,
        help="Limit the number of snapshots used in the model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run optimization even if an output .nc file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_count = args.snapshot_count
    if args.debug and snapshot_count is None:
        snapshot_count = DEFAULT_DEBUG_SNAPSHOT_COUNT

    print("Loading timeseries and building the base electricity network...")
    all_timeseries_data = load_all_country_timeseries(snapshot_count=snapshot_count)
    base_n = build_base_network(all_timeseries_data)

    _, _ = run_baseline(base_n, force=args.force)

    results = [run_scenario(scenario, base_n, force=args.force) for scenario in args.scenarios]

    print("\nComparison summary:")
    for result in results:
        print(
            f"  {result['scenario']}: pipeline {result['pipeline_energy_gwh']:.1f} GWh vs "
            f"electricity interconnectors {result['electricity_interconnector_energy_gwh']:.1f} GWh"
        )

    flow_plot_file = plot_gas_pipeline_flow_networks(OUTPUT_DIR)
    print(f"\nSaved gas pipeline flow network plot: {flow_plot_file}")

    gas_scenario = args.scenarios[0] if args.scenarios else None
    if gas_scenario is not None:
        n_gas = load_saved_pipeline_network(gas_scenario)
        if n_gas is not None:
            n_ch4 = load_saved_pipeline_network("CH4")
            n_h2 = load_saved_pipeline_network("H2")
            route_plot_file = plot_annual_pipeline_route_transport_by_carrier(
                OUTPUT_DIR,
                n=n_gas if n_gas is not None else None,
                n_ch4=n_ch4,
                n_h2=n_h2,
            )
            if route_plot_file is not None:
                print(f"Saved route transport comparison plot: {route_plot_file}")
        else:
            print("Skipping route transport plot because the gas pipeline network could not be loaded.")
    else:
        print("No gas pipeline scenario selected; skipping route transport plot.")

if __name__ == "__main__":
    main()
