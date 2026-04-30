from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETWORK_DIR = PROJECT_ROOT / "results" / "networks"
OUTPUT_DIR = PROJECT_ROOT / "results" / "interconnected"

WEATHER_YEAR = 2016
NETWORK_NAME = f"interconnected_{WEATHER_YEAR}.nc"

REPORT_FONT_SIZE = 14

CARRIER_ALIASES = {
    "solar": ["solar"],
    "onshore wind": ["onwind", "onshore wind", "wind_onshore", "onshore"],
    "offshore wind": ["offwind", "offshore wind", "wind_offshore", "offshore"],
    "CCGT": ["gas CCGT", "CCGT", "ccgt", "gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
    "electricity": ["electricity", "AC"],
    "battery charger": ["battery charger", "battery_charger"],
    "battery discharger": ["battery discharger", "battery_discharge", "battery_discharger"],
}

DEFAULT_CARRIER_COLORS = {
    "electricity": "#4C566A",
    "AC": "#4C566A",
    "solar": "#EBCB3B",
    "onshore wind": "#5AA469",
    "onwind": "#5AA469",
    "offshore wind": "#2E86AB",
    "offwind": "#2E86AB",
    "CCGT": "#D08770",
    "gas": "#D08770",
    "gas CCGT": "#D08770",
    "coal": "#5C5C5C",
    "nuclear": "#8F6BB3",
    "battery charger": "#C06C84",
    "battery discharger": "#6C5B7B",
}

GENERATION_STACK_ORDER = [
    "solar",
    "offshore wind",
    "onshore wind",
    "CCGT",
    "coal",
    "nuclear",
    "battery discharger",
]

ENERGY_BALANCE_ORDER = [
    "electricity",
    "battery charger",
    "solar",
    "offshore wind",
    "onshore wind",
    "CCGT",
    "coal",
    "nuclear",
    "battery discharger",
]

EXCHANGE_COLORS = {
    "DE": "#5AA469",
    "SE": "#D08770",
    "NO": "#2E86AB",
}


# =========================================================
# PLOT STYLE
# =========================================================

def set_report_plot_style() -> None:
    """
    Set basic report-ready matplotlib style.
    """
    plt.rcParams.update(
        {
            "font.size": REPORT_FONT_SIZE,
            "axes.labelsize": REPORT_FONT_SIZE,
            "xtick.labelsize": REPORT_FONT_SIZE,
            "ytick.labelsize": REPORT_FONT_SIZE,
            "legend.fontsize": REPORT_FONT_SIZE,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """
    Save figure as PNG and PDF.
    """
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


# =========================================================
# GENERAL HELPERS
# =========================================================

def load_network(network_file: Path) -> pypsa.Network:
    """
    Load the PyPSA network from a NetCDF file.
    """
    if not network_file.exists():
        raise FileNotFoundError(f"Network file not found: {network_file}")

    network = pypsa.Network(network_file)
    network.sanitize()

    return network


def create_analysis_folder(project_root: Path) -> Path:
    """
    Create the results/interconnected folder if it does not exist.
    """
    folder = project_root / "results" / "interconnected"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def map_carrier_to_display_name(carrier: str) -> str:
    """
    Map internal model carrier names to report-friendly names.
    """
    for display_name, aliases in CARRIER_ALIASES.items():
        if carrier in aliases:
            return display_name

    return carrier


def get_carrier_color(carrier: str) -> str:
    """
    Return report color for a carrier.
    """
    display_name = map_carrier_to_display_name(carrier)

    if display_name in DEFAULT_CARRIER_COLORS:
        return DEFAULT_CARRIER_COLORS[display_name]

    if carrier in DEFAULT_CARRIER_COLORS:
        return DEFAULT_CARRIER_COLORS[carrier]

    return "#999999"


def drop_empty_carriers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all-zero and all-NaN columns.
    """
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, (df.fillna(0.0).abs() > 0.0).any(axis=0)]

    return df


def reorder_columns(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """
    Reorder columns according to a preferred order while keeping remaining columns.
    """
    ordered = [column for column in order if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]

    return df[ordered + remaining]


# =========================================================
# SYSTEM SUMMARY
# =========================================================

def print_system_summary(n: pypsa.Network) -> None:
    """
    Print a short system summary.
    """
    print("System Summary:")
    print(f"- Buses: {len(n.buses)}")
    print(f"- Lines: {len(n.lines)}")
    print(f"- Stores: {len(n.stores)}")
    print(f"- Links: {len(n.links)}")

    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]
    neighbours = set(dk_lines.bus0).union(set(dk_lines.bus1)) - {"DK"}
    print(f"- DK has {len(neighbours)} neighbours: {sorted(neighbours)}")

    cycle_lines = [("DK", "SE"), ("SE", "DE"), ("DE", "DK")]

    has_cycle = all(
        any(
            (
                (line.bus0 == b0 and line.bus1 == b1)
                or (line.bus0 == b1 and line.bus1 == b0)
            )
            for _, line in n.lines.iterrows()
        )
        for b0, b1 in cycle_lines
    )

    print(f"- Closed cycle DK-SE-DE-DK: {'Yes' if has_cycle else 'No'}")


# =========================================================
# TABLE EXPORTS
# =========================================================

def save_line_summary(n: pypsa.Network, folder: Path) -> None:
    """
    Save line_summary.csv with line capacity and utilisation indicators.
    """
    lines = n.lines.copy()
    flows = n.lines_t.p0

    utilisation = flows.abs().div(lines.s_nom, axis=1)

    summary = pd.DataFrame(
        {
            "line_name": lines.index,
            "bus0": lines.bus0,
            "bus1": lines.bus1,
            "s_nom_MW": lines.s_nom,
            "x": lines.x,
            "max_abs_flow_MW": flows.abs().max(),
            "mean_abs_flow_MW": flows.abs().mean(),
            "utilization_max_pu": utilisation.max(),
            "congestion_hours": (utilisation >= 0.99).sum(),
        }
    )

    summary.to_csv(folder / "line_summary.csv", index=False)


def save_dk_battery_kpis(n: pypsa.Network, folder: Path) -> None:
    """
    Save dk_battery_kpis.csv with optimized battery capacities and annual flows.
    """
    required_store = "DK_battery_store"
    required_charger = "DK_battery_charger"
    required_discharger = "DK_battery_discharger"

    missing = []

    if required_store not in n.stores.index:
        missing.append(required_store)
    if required_charger not in n.links.index:
        missing.append(required_charger)
    if required_discharger not in n.links.index:
        missing.append(required_discharger)

    if missing:
        print(
            "Skipping battery KPI export because the following components are missing: "
            f"{missing}"
        )
        return

    energy_capacity = n.stores.loc[required_store, "e_nom_opt"]
    charge_power = n.links.loc[required_charger, "p_nom_opt"]
    discharge_power = n.links.loc[required_discharger, "p_nom_opt"]

    charger_flow = n.links_t.p0[required_charger]
    discharger_flow = n.links_t.p0[required_discharger]

    annual_charged = charger_flow.clip(lower=0).sum()
    annual_discharged = discharger_flow.clip(lower=0).sum()

    kpis = pd.DataFrame(
        {
            "optimized_energy_capacity_MWh": [energy_capacity],
            "optimized_charge_power_MW": [charge_power],
            "optimized_discharge_power_MW": [discharge_power],
            "annual_charged_energy_MWh": [annual_charged],
            "annual_discharged_energy_MWh": [annual_discharged],
        }
    )

    kpis.to_csv(folder / "dk_battery_kpis.csv", index=False)

    print("\nDK Battery KPIs:")
    print(kpis.round(3).to_string(index=False))


def create_line_loading_summary(n: pypsa.Network, folder: Path) -> None:
    """
    Create and save a line loading summary table.
    """
    line_labels = {
        "DK_NO": "DK-NO",
        "DK_SE": "DK-SE",
        "DK_DE": "DK-DE",
        "DE_SE": "DE-SE",
        "NO_SE": "NO-SE",
        "DE_NO": "DE-NO",
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

        summary_data.append(
            {
                "Line": label,
                "Line capacity [MW]": int(round(s_nom)),
                "Congested hours": congested_hours,
                "Share of year [%]": share_of_year_pct,
                "Max loading [%]": max_loading_pct,
            }
        )

    df = pd.DataFrame(summary_data)

    df = df.sort_values(
        by=["Max loading [%]", "Congested hours"],
        ascending=False,
    )

    df.to_csv(folder / "line_loading_summary.csv", index=False)

    print("\nLine Loading Summary:")
    print(df.to_string(index=False))

    latex_str = df.to_latex(
        index=False,
        column_format="lcccc",
        float_format="%.1f",
    )

    with open(folder / "line_loading_summary.tex", "w") as file:
        file.write(latex_str)


def save_data_for_last_step(n: pypsa.Network, folder: Path) -> None:
    """
    Save first-timestep data for the analytical PTDF verification.

    This function exports:
    - PyPSA line flows p0 for the selected line order
    - line capacities and loading
    - nodal injections implied by the PyPSA line flows
    - the incidence matrix used for the manual PTDF calculation

    PyPSA convention:
    p0 > 0 means flow from bus0 to bus1.
    """
    first_ts = n.snapshots[0]

    bus_order = ["DK", "DE", "NO", "SE"]
    line_order = ["DK_DE", "DK_NO", "DK_SE", "DE_NO", "DE_SE", "NO_SE"]

    missing_lines = [line for line in line_order if line not in n.lines.index]
    missing_buses = [bus for bus in bus_order if bus not in n.buses.index]

    if missing_lines:
        raise KeyError(
            f"Missing lines in network: {missing_lines}\n"
            f"Available lines are: {list(n.lines.index)}"
        )

    if missing_buses:
        raise KeyError(
            f"Missing buses in network: {missing_buses}\n"
            f"Available buses are: {list(n.buses.index)}"
        )

    # -----------------------------------------------------
    # Export PyPSA line flows
    # -----------------------------------------------------
    line_rows = []

    for line in line_order:
        bus0 = n.lines.loc[line, "bus0"]
        bus1 = n.lines.loc[line, "bus1"]
        flow = n.lines_t.p0.loc[first_ts, line]
        capacity = n.lines.loc[line, "s_nom"]
        loading = abs(flow) / capacity * 100

        line_rows.append(
            {
                "Line": line.replace("_", "-"),
                "bus0": bus0,
                "bus1": bus1,
                "PyPSA flow p0 [MW]": flow,
                "Line capacity [MW]": capacity,
                "Loading [%]": loading,
            }
        )

    line_df = pd.DataFrame(line_rows)
    line_df.to_csv(folder / "first_timestep_line_flows.csv", index=False)

    print("\nFirst timestep:")
    print(first_ts)

    print("\nPyPSA line flows at first timestep:")
    print(line_df.round(3).to_string(index=False))

    # -----------------------------------------------------
    # Build incidence matrix K with +1 at bus0 and -1 at bus1
    # -----------------------------------------------------
    K = pd.DataFrame(
        0.0,
        index=bus_order,
        columns=[line.replace("_", "-") for line in line_order],
    )

    for line in line_order:
        label = line.replace("_", "-")
        bus0 = n.lines.loc[line, "bus0"]
        bus1 = n.lines.loc[line, "bus1"]

        K.loc[bus0, label] = 1.0
        K.loc[bus1, label] = -1.0

    flow_vector = line_df.set_index("Line")["PyPSA flow p0 [MW]"]
    implied_injection = K @ flow_vector

    bus_df = pd.DataFrame(
        {
            "Bus": implied_injection.index,
            "Nodal injection implied by line flows [MW]": implied_injection.values,
        }
    )

    bus_df.to_csv(folder / "first_timestep_bus_injections_from_flows.csv", index=False)
    K.to_csv(folder / "first_timestep_incidence_matrix.csv")

    print("\nNodal injections implied by PyPSA line flows:")
    print(bus_df.round(3).to_string(index=False))

    # -----------------------------------------------------
    # Optional: PTDF verification inside Python
    # -----------------------------------------------------
    x = 0.1
    b = 1.0 / x

    K_np = K.values
    B_l = np.eye(len(line_order)) * b

    slack_bus = "DK"
    non_slack_buses = [bus for bus in bus_order if bus != slack_bus]

    K_red = K.loc[non_slack_buses].values
    L_red = K_red @ B_l @ K_red.T
    L_red_inv = np.linalg.inv(L_red)

    p_red = bus_df.set_index("Bus").loc[non_slack_buses, "Nodal injection implied by line flows [MW]"].values

    # With K convention (+1 at bus0, -1 at bus1), this gives PyPSA p0 sign convention.
    theta_red = L_red_inv @ p_red
    manual_flows = B_l @ K_red.T @ theta_red

    verification_df = line_df[["Line", "PyPSA flow p0 [MW]"]].copy()
    verification_df["Manual PTDF flow [MW]"] = manual_flows
    verification_df["Difference [MW]"] = (
        verification_df["Manual PTDF flow [MW]"]
        - verification_df["PyPSA flow p0 [MW]"]
    )

    verification_df.to_csv(folder / "first_timestep_ptdf_verification.csv", index=False)

    print("\nPTDF verification using PyPSA-implied nodal injections:")
    print(verification_df.round(6).to_string(index=False))

    print(
        "\nSaved first-timestep PTDF verification data to:\n"
        f"- {folder / 'first_timestep_line_flows.csv'}\n"
        f"- {folder / 'first_timestep_bus_injections_from_flows.csv'}\n"
        f"- {folder / 'first_timestep_incidence_matrix.csv'}\n"
        f"- {folder / 'first_timestep_ptdf_verification.csv'}"
    )


# =========================================================
# ENERGY BALANCE FOR DENMARK
# =========================================================

def get_dk_energy_balance_by_carrier(n: pypsa.Network) -> pd.DataFrame:
    """
    Return time-resolved Denmark energy balance by carrier.

    Positive values are supply to the DK electricity bus.
    Negative values are consumption from the DK electricity bus.
    """
    try:
        balance = n.statistics.energy_balance(
            aggregate_time=False,
            nice_names=False,
        )

        if "bus" in balance.index.names:
            balance_dk = balance.xs("DK", level="bus")
            balance_by_carrier = balance_dk.groupby(level="carrier").sum()
            balance_by_carrier_t = balance_by_carrier.T

            if len(balance_by_carrier_t.index) == len(n.snapshots):
                balance_by_carrier_t.index = n.snapshots

            balance_by_carrier_t = balance_by_carrier_t.rename(
                columns=lambda c: map_carrier_to_display_name(c)
            )
            balance_by_carrier_t = balance_by_carrier_t.T.groupby(level=0).sum().T
            balance_by_carrier_t = drop_empty_carriers(balance_by_carrier_t)
            balance_by_carrier_t = reorder_columns(
                balance_by_carrier_t,
                ENERGY_BALANCE_ORDER,
            )

            print("Using n.statistics.energy_balance() for DK energy balance.")
            return balance_by_carrier_t

    except Exception as error:
        print(
            "Could not extract DK balance directly from n.statistics.energy_balance(). "
            f"Falling back to component-based balance. Reason: {error}"
        )

    print(
        "Using component-based DK energy balance because the statistics output "
        "does not expose a usable bus-level index."
    )

    balance_by_carrier_t = pd.DataFrame(index=n.snapshots)

    dk_loads = n.loads[n.loads.bus == "DK"].index
    if len(dk_loads) > 0:
        balance_by_carrier_t["electricity"] = -n.loads_t.p_set[dk_loads].sum(axis=1)

    dk_generators = n.generators[n.generators.bus == "DK"]

    for generator in dk_generators.index:
        carrier = map_carrier_to_display_name(n.generators.at[generator, "carrier"])

        if carrier not in balance_by_carrier_t.columns:
            balance_by_carrier_t[carrier] = 0.0

        balance_by_carrier_t[carrier] = balance_by_carrier_t[carrier].add(
            n.generators_t.p[generator],
            fill_value=0.0,
        )

    if "DK_battery_charger" in n.links.index:
        balance_by_carrier_t["battery charger"] = -n.links_t.p0[
            "DK_battery_charger"
        ].clip(lower=0.0)

    if "DK_battery_discharger" in n.links.index:
        balance_by_carrier_t["battery discharger"] = n.links_t.p0[
            "DK_battery_discharger"
        ].clip(lower=0.0)

    balance_by_carrier_t = drop_empty_carriers(balance_by_carrier_t)
    balance_by_carrier_t = reorder_columns(
        balance_by_carrier_t,
        ENERGY_BALANCE_ORDER,
    )

    return balance_by_carrier_t


# =========================================================
# PLOTS
# =========================================================

def plot_interconnector_summary(n: pypsa.Network, folder: Path) -> None:
    """
    Create interconnector_summary.png and save the underlying data.
    """
    preferred_lines = ["DK_NO", "DK_SE", "DK_DE"]
    lines = [line for line in preferred_lines if line in n.lines.index]

    if not lines:
        print("Skipping interconnector summary because no DK interconnector lines were found.")
        return

    data = []

    for line in lines:
        flow = n.lines_t.p0[line]
        s_nom = n.lines.loc[line, "s_nom"]
        bus0 = n.lines.loc[line, "bus0"]

        if bus0 == "DK":
            export_mwh = flow.clip(lower=0).sum()
            import_mwh = (-flow).clip(lower=0).sum()
        else:
            import_mwh = flow.clip(lower=0).sum()
            export_mwh = (-flow).clip(lower=0).sum()

        loading = flow.abs() / s_nom
        near_cap_hours = int((loading >= 0.99).sum())
        total_hours = len(flow)
        share_pct = round(100 * near_cap_hours / total_hours, 1)

        data.append(
            {
                "Line": line.replace("_", "-"),
                "Annual imports to DK [GWh]": import_mwh / 1000,
                "Annual exports from DK [GWh]": export_mwh / 1000,
                "Line capacity [GW]": s_nom / 1000,
                "Share of year near capacity [%]": share_pct,
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(folder / "interconnector_summary.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.4))

    x = np.arange(len(lines))
    width = 0.35

    ax1.bar(
        x - width / 2,
        df["Annual imports to DK [GWh]"],
        width,
        label="Imports to DK",
        color="#2E86AB",
        alpha=0.9,
    )

    ax1.bar(
        x + width / 2,
        df["Annual exports from DK [GWh]"],
        width,
        label="Exports from DK",
        color="#D08770",
        alpha=0.9,
    )

    ax1.set_xlabel("Interconnector")
    ax1.set_ylabel("Annual energy [GWh]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Line"])
    ax1.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["Line capacity [GW]"],
        color="black",
        linewidth=2.0,
        marker="o",
        label="Line capacity",
    )

    ax2.set_ylabel("Line capacity [GW]")
    ax2.tick_params(axis="y", labelsize=REPORT_FONT_SIZE)

    bars = ax1.get_legend_handles_labels()
    line_legend = ax2.get_legend_handles_labels()

    ax1.legend(
        bars[0] + line_legend[0],
        bars[1] + line_legend[1],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, folder / "interconnector_summary.png")
    plt.close(fig)


def plot_interconnector_utilisation_comparison(n: pypsa.Network, folder: Path) -> None:
    """
    Create monthly interconnector utilisation comparison.
    """
    monthly_data = {}

    for line in n.lines.index:
        flow = n.lines_t.p0[line]
        s_nom = n.lines.loc[line, "s_nom"]

        loading = flow.abs() / s_nom * 100
        monthly_avg = loading.groupby(flow.index.month).mean()

        monthly_data[line.replace("_", "-")] = monthly_avg

    df = pd.DataFrame(monthly_data)
    df.index.name = "Month"

    df.to_csv(folder / "interconnector_utilisation_comparison.csv")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    x = np.arange(1, 13)

    for interconnector in df.columns:
        ax.plot(
            x,
            df[interconnector],
            marker="o",
            linewidth=2.0,
            label=interconnector,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Utilisation [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, folder / "interconnector_utilisation_comparison.png")
    plt.close(fig)


def analyze_denmark_export_origin(n: pypsa.Network, folder: Path) -> None:
    """
    Analyze the origin of Denmark's exports: local surplus versus re-exported imports.
    """
    dk_gens = n.generators[n.generators.bus == "DK"]
    dk_generation = n.generators_t.p[dk_gens.index].sum(axis=1)

    if "DK_electricity_demand" not in n.loads_t.p_set.columns:
        print("Skipping Denmark export-origin analysis because DK demand was not found.")
        return

    dk_load = n.loads_t.p_set["DK_electricity_demand"]

    battery_charge = pd.Series(0.0, index=n.snapshots)
    battery_discharge = pd.Series(0.0, index=n.snapshots)

    if "DK_battery_charger" in n.links.index:
        battery_charge = n.links_t.p0["DK_battery_charger"].clip(lower=0)

    if "DK_battery_discharger" in n.links.index:
        battery_discharge = n.links_t.p0["DK_battery_discharger"].clip(lower=0)

    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]

    imports = pd.Series(0.0, index=n.snapshots)
    exports = pd.Series(0.0, index=n.snapshots)

    for line in dk_lines.index:
        flow = n.lines_t.p0[line]

        if n.lines.loc[line, "bus0"] == "DK":
            exports += flow.clip(lower=0)
            imports += (-flow).clip(lower=0)
        else:
            imports += flow.clip(lower=0)
            exports += (-flow).clip(lower=0)

    reexport = pd.concat([exports, imports], axis=1).min(axis=1)
    local_export = (exports - imports).clip(lower=0)

    monthly_data = []

    for month in range(1, 13):
        mask = n.snapshots.month == month

        monthly_data.append(
            {
                "Month": month,
                "Exports total [GWh]": exports[mask].sum() / 1000,
                "Exports from DK surplus [GWh]": local_export[mask].sum() / 1000,
                "Re-exported imports [GWh]": reexport[mask].sum() / 1000,
                "Imports [GWh]": imports[mask].sum() / 1000,
                "DK generation [GWh]": dk_generation[mask].sum() / 1000,
                "DK demand [GWh]": dk_load[mask].sum() / 1000,
                "Battery charge [GWh]": battery_charge[mask].sum() / 1000,
                "Battery discharge [GWh]": battery_discharge[mask].sum() / 1000,
            }
        )

    df = pd.DataFrame(monthly_data)

    df["Check"] = (
        df["Exports total [GWh]"]
        - df["Exports from DK surplus [GWh]"]
        - df["Re-exported imports [GWh]"]
    )

    print("\nDenmark Export Origin Monthly Summary:")
    print(df.to_string(index=False))

    df_save = df.drop(columns=["Check"])
    df_save.to_csv(folder / "denmark_export_origin.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    x = np.arange(1, 13)

    ax.bar(
        x,
        df["Exports from DK surplus [GWh]"],
        label="Exports from DK surplus",
        color="#D08770",
        alpha=0.9,
    )

    ax.bar(
        x,
        df["Re-exported imports [GWh]"],
        bottom=df["Exports from DK surplus [GWh]"],
        label="Re-exported imports",
        color="#EBCB8B",
        alpha=0.9,
    )

    ax.plot(
        x,
        df["Exports total [GWh]"],
        color="black",
        linewidth=2.0,
        marker="o",
        label="Total exports",
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Exports [GWh]")
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, folder / "denmark_export_origin.png")
    plt.close(fig)


def save_generation_summary_by_country(n: pypsa.Network, folder: Path) -> None:
    """
    Save and plot optimized generation capacity by country and technology.
    """
    rows = []

    for gen in n.generators.index:
        bus = n.generators.loc[gen, "bus"]
        carrier = n.generators.loc[gen, "carrier"]

        if "p_nom_opt" in n.generators.columns:
            capacity = n.generators.loc[gen, "p_nom_opt"]
        else:
            capacity = n.generators.loc[gen, "p_nom"]

        annual_generation_mwh = n.generators_t.p[gen].sum()
        annual_generation_gwh = annual_generation_mwh / 1000

        rows.append(
            {
                "Country": bus,
                "Generator": gen,
                "Carrier": carrier,
                "Optimized capacity [MW]": capacity,
                "Annual generation [GWh]": annual_generation_gwh,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(folder / "generation_by_generator_and_country.csv", index=False)

    df_by_country_carrier = df.groupby(
        ["Country", "Carrier"],
        as_index=False,
    ).agg(
        {
            "Optimized capacity [MW]": "sum",
            "Annual generation [GWh]": "sum",
        }
    )

    df_by_country_carrier.to_csv(
        folder / "generation_by_country_and_carrier.csv",
        index=False,
    )

    print("\nGeneration Summary by Country and Carrier:")
    print(df_by_country_carrier.to_string(index=False))

    plot_df = df_by_country_carrier.pivot(
        index="Country",
        columns="Carrier",
        values="Optimized capacity [MW]",
    ).fillna(0)

    preferred_country_order = [c for c in ["DK", "DE", "SE", "NO"] if c in plot_df.index]
    remaining_countries = [c for c in plot_df.index if c not in preferred_country_order]
    plot_df = plot_df.loc[preferred_country_order + remaining_countries]

    preferred_tech_order = [
        "coal",
        "gas",
        "gas CCGT",
        "CCGT",
        "nuclear",
        "offwind",
        "onwind",
        "solar",
    ]

    tech_order = [t for t in preferred_tech_order if t in plot_df.columns]
    remaining_tech = [t for t in plot_df.columns if t not in tech_order]
    plot_df = plot_df[tech_order + remaining_tech]

    colors = [get_carrier_color(col) for col in plot_df.columns]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    plot_df.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.75,
        color=colors,
    )

    ax.set_xlabel("Country")
    ax.set_ylabel("Optimized capacity [MW]")
    ax.tick_params(axis="x", labelrotation=0, labelsize=REPORT_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=REPORT_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        ncol=2,
    )

    fig.tight_layout()
    save_figure(fig, folder / "generation_capacity_by_country_and_technology.png")
    plt.close(fig)


def plot_denmark_dispatch_strategy(n: pypsa.Network, folder: Path) -> None:
    """
    Plot Denmark's dispatch strategy for the winter week with the highest
    average Danish electricity demand.

    The upper panel uses the Denmark electricity-bus energy balance by carrier.
    The lower panel shows imports and exports with neighbouring countries.

    Positive exchange values mean imports to Denmark.
    Negative exchange values mean exports from Denmark.
    """
    snapshots = n.snapshots
    winter_mask = (snapshots.month == 12) | (snapshots.month == 1)

    if "DK_electricity_demand" not in n.loads_t.p_set.columns:
        raise KeyError(
            "Could not find load 'DK_electricity_demand' in n.loads_t.p_set. "
            f"Available loads are: {list(n.loads_t.p_set.columns)}"
        )

    dk_load = n.loads_t.p_set["DK_electricity_demand"]
    winter_load = dk_load.loc[winter_mask]

    weekly_avg_load = winter_load.resample("W").mean()
    max_load_week_end = weekly_avg_load.idxmax()

    week_start = max_load_week_end - pd.Timedelta(days=6)
    week_end = max_load_week_end

    week_index = dk_load.loc[week_start:week_end].index
    week_load = dk_load.loc[week_index]

    dk_balance = get_dk_energy_balance_by_carrier(n)
    week_balance = dk_balance.loc[week_index].copy()
    week_balance = reorder_columns(week_balance, ENERGY_BALANCE_ORDER)

    generation_columns = [
        carrier for carrier in GENERATION_STACK_ORDER if carrier in week_balance.columns
    ]

    generation_stack = week_balance[generation_columns].clip(lower=0.0)
    generation_stack = drop_empty_carriers(generation_stack)
    generation_stack = reorder_columns(generation_stack, GENERATION_STACK_ORDER)

    battery_charge = pd.Series(0.0, index=week_index)

    if "battery charger" in week_balance.columns:
        battery_charge = (-week_balance["battery charger"]).clip(lower=0.0)

    dk_lines = n.lines[(n.lines.bus0 == "DK") | (n.lines.bus1 == "DK")]

    exchanges = {}

    for line in dk_lines.index:
        bus0 = n.lines.at[line, "bus0"]
        bus1 = n.lines.at[line, "bus1"]

        neighbour = bus1 if bus0 == "DK" else bus0
        flow = n.lines_t.p0[line].loc[week_index]

        if bus0 == "DK":
            dk_exchange = -flow
        else:
            dk_exchange = flow

        if neighbour not in exchanges:
            exchanges[neighbour] = pd.Series(0.0, index=week_index)

        exchanges[neighbour] = exchanges[neighbour].add(
            dk_exchange,
            fill_value=0.0,
        )

    output_data = pd.DataFrame(index=week_index)
    output_data["DK demand [MW]"] = week_load
    output_data["Battery charge [MW]"] = battery_charge

    for carrier in week_balance.columns:
        output_data[f"DK balance {carrier} [MW]"] = week_balance[carrier]

    for neighbour, series in exchanges.items():
        output_data[f"DK exchange with {neighbour} [MW]"] = series

    output_data.to_csv(folder / "denmark_dispatch_strategy_winter_week.csv")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    bottom = np.zeros(len(week_index))

    for carrier in generation_stack.columns:
        series = generation_stack[carrier].fillna(0.0).values

        ax1.fill_between(
            week_index,
            bottom,
            bottom + series,
            label=carrier,
            color=get_carrier_color(carrier),
            alpha=0.9,
            linewidth=0.0,
        )

        bottom += series

    if battery_charge.abs().sum() > 0:
        ax1.plot(
            week_index,
            week_load + battery_charge,
            color="grey",
            linewidth=1.8,
            linestyle="--",
            label="Demand incl. battery charging",
        )

    ax1.plot(
        week_index,
        week_load,
        color="black",
        linewidth=2.2,
        label="DK demand",
    )

    ax1.set_ylabel("Power [MW]")
    ax1.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    for neighbour, series in exchanges.items():
        ax2.plot(
            week_index,
            series,
            linewidth=2.0,
            label=neighbour,
            color=EXCHANGE_COLORS.get(neighbour, None),
        )

    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_ylabel("Exchange [MW]")
    ax2.set_xlabel("Time")
    ax2.tick_params(axis="both", labelsize=REPORT_FONT_SIZE)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )

    ax2.legend(
        loc="upper right",
        ncol=3,
        frameon=True,
        framealpha=0.95,
    )

    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()

    outfile = folder / "denmark_dispatch_strategy_winter_week.png"
    save_figure(fig, outfile)
    plt.close(fig)

    print(f"Saved Denmark dispatch strategy plot to {outfile}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Main function to run the analysis.
    """
    set_report_plot_style()

    network_file = NETWORK_DIR / NETWORK_NAME

    n = load_network(network_file)
    folder = create_analysis_folder(PROJECT_ROOT)

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

    print(f"\nAnalysis complete. Files saved in {folder}")


if __name__ == "__main__":
    main()