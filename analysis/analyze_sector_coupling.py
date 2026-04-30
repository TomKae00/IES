# =========================================================
# TASK i) INTERCONNECTED SECTOR COUPLING PATCH
# =========================================================
# This corrected version keeps the interconnected multi-country model from d),
# then adds heat-sector coupling and a system-wide CO2 cap for task i).
#
# Copy this into your model file as follows:
# 1) Put the SETTINGS block near the top of the file, after ACTIVE_SCENARIO.
# 2) Add apply_capacity_caps() before create_network().
# 3) Add the plotting/helper functions in the PLOTS section.
# 4) Replace your main() with the main() at the bottom.

from pathlib import Path
import sys
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt
import pypsa
import numpy as np

# Because this file is in analysis/, add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.helpers import (
    silence_gurobi_logger,
    prepare_costs,
    load_all_countries_timeseries,
    load_heat_timeseries,
    calculate_conventional_marginal_cost,
)

from model.scenarios import (
    FINANCIAL_PARAMETERS,
    FILE_PATHS,
    SCENARIOS,
)

# Import the original model-construction functions.
# If your original file is not model/model.py, change this import to match the file
# that contains def create_network(...), def optimize_and_save_network(...), and
# def print_model_summary(...).
from model.model import (
    create_network,
    optimize_and_save_network,
    print_model_summary,
)

# ---------------------------------------------------------
# SETTINGS FOR TASK i
# ---------------------------------------------------------

ACTIVE_SCENARIO = "sector_coupling"

# Use None to keep the countries from SCENARIOS[ACTIVE_SCENARIO].
# If your base scenario is not already the interconnected model from d), set:
# TASK_I_COUNTRIES = ["DK", "NO", "SE", "DE"]
TASK_I_COUNTRIES = None

# System-wide CO2 allowance for all modelled countries and sectors.
# Unit: tonnes CO2 over the modelled weather year.
SYSTEM_CO2_CAP_TCO2 = 5_000_000.0

# Optional country-specific technology caps.
# Use these only if you want to analyse a specific cap, e.g. Danish offshore wind.
# Leave entries out or set to None if no cap should apply.
GENERATOR_CAPS_MW = {
    # "DK_offshore_wind": 2650.0,
    # "DK_onshore_wind": 0.0,
    # "DE_nuclear": 0.0,
}

LINK_CAPS_MW = {
    # "DK_ASHP": 5000.0,
}

STORE_CAPS_MWH = {
    # "DK_battery_store": 10000.0,
}

# Denmark has no nuclear power plants. If nuclear exists in the cost data,
# the original add_electricity() function would otherwise allow DK_nuclear.
DISABLE_DK_NUCLEAR = True


def build_task_i_interconnected_sector_coupling_scenario(base_scenario: dict) -> dict:
    """
    Build the corrected task i scenario.

    Interpretation of the assignment:
    - Start from the interconnected model from d).
    - Add at least one additional sector; here: decentral heat.
    - Co-optimise electricity, heat, storage, and optional gas/H2 networks.
    - Apply one system-wide CO2 cap.

    This does NOT reduce the model to Denmark only. Denmark-specific analysis can
    still be shown in the plots, but the optimisation remains multi-country.
    """

    scenario = deepcopy(base_scenario)

    countries = scenario.get("countries", []) if TASK_I_COUNTRIES is None else TASK_I_COUNTRIES

    scenario.update(
        {
            "name": "task_i_interconnected_sector_coupling_CO2_cap",
            "countries": countries,
            "with_interconnectors": True,
            "with_heat_sector": True,
            "with_heat_storage": True,
            "with_battery_storage": True,
            "co2_limit": SYSTEM_CO2_CAP_TCO2,
            # Keep the base scenario gas/H2 choices. If your task i should only
            # couple electricity and heat, set these two to False manually.
            "with_ch4_network": scenario.get("with_ch4_network", False),
            "with_h2_network": scenario.get("with_h2_network", False),
            "co2_price": scenario.get("co2_price", 0.0),
        }
    )

    generator_caps = {
        gen: cap for gen, cap in GENERATOR_CAPS_MW.items() if cap is not None
    }
    link_caps = {
        link: cap for link, cap in LINK_CAPS_MW.items() if cap is not None
    }
    store_caps = {
        store: cap for store, cap in STORE_CAPS_MWH.items() if cap is not None
    }

    if DISABLE_DK_NUCLEAR and "DK" in countries:
        generator_caps["DK_nuclear"] = 0.0

    scenario["capacity_caps"] = {
        "generators": generator_caps,
        "links": link_caps,
        "stores": store_caps,
    }

    return scenario


def apply_capacity_caps(n: pypsa.Network, scenario: dict) -> None:
    """
    Apply optional p_nom/e_nom caps after all components are created, but before
    optimisation. This avoids editing each n.add(...) call separately.
    """

    caps = scenario.get("capacity_caps", {})

    for gen, cap in caps.get("generators", {}).items():
        if gen in n.generators.index:
            n.generators.at[gen, "p_nom_max"] = cap
        else:
            print(f"Warning: generator cap skipped because {gen} is not in n.generators")

    for link, cap in caps.get("links", {}).items():
        if link in n.links.index:
            n.links.at[link, "p_nom_max"] = cap
        else:
            print(f"Warning: link cap skipped because {link} is not in n.links")

    for store, cap in caps.get("stores", {}).items():
        if store in n.stores.index:
            n.stores.at[store, "e_nom_max"] = cap
        else:
            print(f"Warning: store cap skipped because {store} is not in n.stores")


# ---------------------------------------------------------
# ADD THIS INSIDE create_network(), just before the CO2 constraint:
# ---------------------------------------------------------
#
#     apply_capacity_caps(n, scenario)
#
#     if scenario["co2_limit"] is not None:
#         add_global_co2_constraint(
#             n=n,
#             co2_limit=scenario["co2_limit"],
#         )
#
# ---------------------------------------------------------


# =========================================================
# TASK i PLOT HELPERS
# =========================================================

DISPLAY_NAMES = {
    "solar": "Solar PV",
    "onwind": "Onshore wind",
    "offwind": "Offshore wind",
    "gas CCGT": "Gas CCGT",
    "CCGT": "Gas CCGT",
    "coal": "Coal",
    "nuclear": "Nuclear",
    "battery": "Battery storage",
    "battery charger": "Battery charger",
    "battery discharger": "Battery discharge",
    "heat pump": "Heat pump",
    "resistive heater": "Resistive heater",
    "gas boiler": "Gas boiler",
    "heat storage": "Heat storage",
    "heat storage charger": "Heat storage charge",
    "heat storage discharger": "Heat storage discharge",
    "CH4 supply": "CH4 supply",
    "CH4 pipeline": "CH4 pipeline",
    "H2 pipeline": "H2 pipeline",
    "electrolysis": "Electrolysis",
    "H2 turbine": "H2 turbine",
    "H2 fuel cell": "H2 fuel cell",
}


def nice_label(x: str) -> str:
    return DISPLAY_NAMES.get(str(x), str(x).replace("_", " "))


def get_country(component_name: str, countries: list[str]) -> str | None:
    """Extract country code from component names such as DK_solar or DK_ASHP."""
    for country in countries:
        if str(component_name) == country or str(component_name).startswith(f"{country}_"):
            return country
    return None


def weighted_sum(n: pypsa.Network, series: pd.Series) -> float:
    """Return weighted annual energy in MWh."""
    weights = n.snapshot_weightings.generators.reindex(series.index).fillna(1.0)
    return float(series.multiply(weights, axis=0).sum())


def link_output_to_bus1(n: pypsa.Network, link: str) -> pd.Series:
    """Positive output from a link into bus1."""
    if link not in n.links.index or link not in n.links_t.p1.columns:
        return pd.Series(0.0, index=n.snapshots)
    return (-n.links_t.p1[link]).clip(lower=0)


def link_input_from_bus0(n: pypsa.Network, link: str) -> pd.Series:
    """Positive input consumed by a link from bus0."""
    if link not in n.links.index or link not in n.links_t.p0.columns:
        return pd.Series(0.0, index=n.snapshots)
    return n.links_t.p0[link].clip(lower=0)


def get_color(n: pypsa.Network, carrier: str) -> str:
    if carrier in n.carriers.index and "color" in n.carriers.columns:
        color = n.carriers.at[carrier, "color"]
        if pd.notna(color):
            return color
    return "#999999"


def clean_series(s: pd.Series, tolerance: float = 1e-6) -> pd.Series:
    s = s.copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s[s.abs() > tolerance]


def country_order_from_network(n: pypsa.Network) -> list[str]:
    countries = []
    for bus in n.buses.index:
        if "_" not in bus and n.buses.at[bus, "carrier"] == "AC":
            countries.append(bus)
    return countries


# =========================================================
# TASK i PLOTS: SYSTEM-WIDE RESULTS
# =========================================================


def plot_task_i_generation_capacity_by_country(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot optimal electricity generation capacity by country and technology.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    records = []

    for gen in n.generators.index:
        country = get_country(gen, countries)
        if country is None:
            continue
        if n.generators.at[gen, "bus"] != country:
            continue
        capacity = n.generators.at[gen, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else 0.0
        records.append(
            {
                "country": country,
                "carrier": n.generators.at[gen, "carrier"],
                "capacity_GW": capacity / 1e3,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        print("Skipping generation capacity plot: no generator capacities found.")
        return

    table = df.pivot_table(index="country", columns="carrier", values="capacity_GW", aggfunc="sum").fillna(0.0)
    table = table.loc[:, table.sum() > 1e-6]

    colors = [get_color(n, c) for c in table.columns]
    ax = table.plot(kind="bar", stacked=True, figsize=(9, 5), color=colors)
    ax.set_ylabel("Installed capacity [GW]")
    ax.set_xlabel("")
    ax.set_title("Optimal electricity generation capacity by country")
    ax.grid(axis="y", alpha=0.3)
    ax.legend([nice_label(c) for c in table.columns], title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "task_i_generation_capacity_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_i_coupling_capacity_by_country(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot heat-sector and flexibility conversion capacities by country.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    relevant_carriers = [
        "heat pump",
        "resistive heater",
        "gas boiler",
        "heat storage charger",
        "heat storage discharger",
        "battery charger",
        "battery discharger",
        "electrolysis",
        "H2 turbine",
        "H2 fuel cell",
        "CCGT",
    ]

    records = []
    for link in n.links.index:
        country = get_country(link, countries)
        if country is None:
            continue
        carrier = n.links.at[link, "carrier"]
        if carrier not in relevant_carriers:
            continue
        capacity = n.links.at[link, "p_nom_opt"] if "p_nom_opt" in n.links.columns else 0.0
        records.append({"country": country, "carrier": carrier, "capacity_GW": capacity / 1e3})

    df = pd.DataFrame(records)
    if df.empty:
        print("Skipping coupling capacity plot: no relevant link capacities found.")
        return

    table = df.pivot_table(index="country", columns="carrier", values="capacity_GW", aggfunc="sum").fillna(0.0)
    table = table.loc[:, table.sum() > 1e-6]

    colors = [get_color(n, c) for c in table.columns]
    ax = table.plot(kind="bar", stacked=True, figsize=(10, 5), color=colors)
    ax.set_ylabel("Installed capacity [GW]")
    ax.set_xlabel("")
    ax.set_title("Sector-coupling and flexibility capacity by country")
    ax.grid(axis="y", alpha=0.3)
    ax.legend([nice_label(c) for c in table.columns], title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "task_i_coupling_capacity_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_i_storage_capacity_by_country(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot battery and heat storage energy capacity by country.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    records = []

    for store in n.stores.index:
        country = get_country(store, countries)
        if country is None:
            continue
        capacity = n.stores.at[store, "e_nom_opt"] if "e_nom_opt" in n.stores.columns else 0.0
        records.append(
            {
                "country": country,
                "carrier": n.stores.at[store, "carrier"],
                "energy_capacity_GWh": capacity / 1e3,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        print("Skipping storage capacity plot: no stores found.")
        return

    table = df.pivot_table(index="country", columns="carrier", values="energy_capacity_GWh", aggfunc="sum").fillna(0.0)
    table = table.loc[:, table.sum() > 1e-6]

    if table.empty:
        print("Skipping storage capacity plot: no storage capacity built.")
        return

    colors = [get_color(n, c) for c in table.columns]
    ax = table.plot(kind="bar", stacked=True, figsize=(8, 5), color=colors)
    ax.set_ylabel("Energy capacity [GWh]")
    ax.set_xlabel("")
    ax.set_title("Storage energy capacity by country")
    ax.grid(axis="y", alpha=0.3)
    ax.legend([nice_label(c) for c in table.columns], title="Storage", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "task_i_storage_capacity_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_i_annual_electricity_generation_by_country(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot annual electricity generation by country and technology.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    records = []

    for gen in n.generators.index:
        country = get_country(gen, countries)
        if country is None:
            continue
        if n.generators.at[gen, "bus"] != country:
            continue
        carrier = n.generators.at[gen, "carrier"]
        energy = weighted_sum(n, n.generators_t.p[gen].clip(lower=0)) / 1e6
        records.append({"country": country, "carrier": carrier, "energy_TWh": energy})

    for link in n.links.index:
        bus1 = n.links.at[link, "bus1"]
        if bus1 not in countries:
            continue
        carrier = n.links.at[link, "carrier"]
        energy = weighted_sum(n, link_output_to_bus1(n, link)) / 1e6
        records.append({"country": bus1, "carrier": carrier, "energy_TWh": energy})

    df = pd.DataFrame(records)
    if df.empty:
        print("Skipping annual electricity plot: no energy found.")
        return

    table = df.pivot_table(index="country", columns="carrier", values="energy_TWh", aggfunc="sum").fillna(0.0)
    table = table.loc[:, table.sum() > 1e-6]

    colors = [get_color(n, c) for c in table.columns]
    ax = table.plot(kind="bar", stacked=True, figsize=(10, 5), color=colors)
    ax.set_ylabel("Annual electricity generation [TWh]")
    ax.set_xlabel("")
    ax.set_title("Annual electricity generation by country")
    ax.grid(axis="y", alpha=0.3)
    ax.legend([nice_label(c) for c in table.columns], title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "task_i_annual_electricity_generation_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_i_annual_heat_supply_by_country(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot annual heat supply by country and technology.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    records = []

    for country in countries:
        heat_bus = f"{country}_heat"

        for link in n.links.index:
            if n.links.at[link, "bus1"] != heat_bus:
                continue
            carrier = n.links.at[link, "carrier"]
            energy = weighted_sum(n, link_output_to_bus1(n, link)) / 1e6
            records.append({"country": country, "carrier": carrier, "heat_TWh": energy})

        # If CH4 network is disabled, gas boiler is represented as a heat generator.
        gas_boiler = f"{country}_gas_boiler"
        if gas_boiler in n.generators.index:
            energy = weighted_sum(n, n.generators_t.p[gas_boiler].clip(lower=0)) / 1e6
            records.append({"country": country, "carrier": "gas boiler", "heat_TWh": energy})

    df = pd.DataFrame(records)
    if df.empty:
        print("Skipping annual heat plot: no heat sector found.")
        return

    table = df.pivot_table(index="country", columns="carrier", values="heat_TWh", aggfunc="sum").fillna(0.0)
    table = table.loc[:, table.sum() > 1e-6]

    colors = [get_color(n, c) for c in table.columns]
    ax = table.plot(kind="bar", stacked=True, figsize=(10, 5), color=colors)
    ax.set_ylabel("Annual heat supply [TWh]")
    ax.set_xlabel("")
    ax.set_title("Annual heat supply by country")
    ax.grid(axis="y", alpha=0.3)
    ax.legend([nice_label(c) for c in table.columns], title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "task_i_annual_heat_supply_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_i_electricity_use_by_country(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot annual electricity use split into base demand and sector-coupling demand.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    records = []

    for country in countries:
        load_name = f"{country}_electricity_demand"
        if load_name in n.loads.index:
            records.append(
                {
                    "country": country,
                    "use": "Base electricity demand",
                    "energy_TWh": weighted_sum(n, n.loads_t.p_set[load_name]) / 1e6,
                }
            )

        for link in n.links.index:
            if n.links.at[link, "bus0"] != country:
                continue
            carrier = n.links.at[link, "carrier"]
            if carrier in ["battery charger", "heat pump", "resistive heater", "electrolysis"]:
                energy = weighted_sum(n, link_input_from_bus0(n, link)) / 1e6
                records.append({"country": country, "use": nice_label(carrier), "energy_TWh": energy})

    df = pd.DataFrame(records)
    if df.empty:
        print("Skipping electricity use plot: no electricity use found.")
        return

    table = df.pivot_table(index="country", columns="use", values="energy_TWh", aggfunc="sum").fillna(0.0)
    table = table.loc[:, table.sum() > 1e-6]

    ax = table.plot(kind="bar", stacked=True, figsize=(10, 5))
    ax.set_ylabel("Annual electricity use [TWh]")
    ax.set_xlabel("")
    ax.set_title("Electricity demand increase from sector coupling")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Use", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "task_i_electricity_use_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_i_annual_net_imports_by_country(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot annual net electricity imports by country.

    Positive values mean the country is a net importer over the modelled year.
    Negative values mean the country is a net exporter.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    net_imports = pd.Series(0.0, index=countries)

    if n.lines.empty:
        print("Skipping net import plot: no interconnectors.")
        return

    for line in n.lines.index:
        bus0 = n.lines.at[line, "bus0"]
        bus1 = n.lines.at[line, "bus1"]
        if bus0 not in countries or bus1 not in countries:
            continue

        flow_p0 = n.lines_t.p0[line]
        annual_p0 = weighted_sum(n, flow_p0) / 1e6

        # p0 positive means export from bus0 to bus1.
        net_imports.loc[bus0] -= annual_p0
        net_imports.loc[bus1] += annual_p0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    net_imports.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("Net imports [TWh/year]")
    ax.set_xlabel("")
    ax.set_title("Annual net electricity imports by country")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "task_i_annual_net_imports_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# TASK i PLOTS: CO2 AND DETAIL WEEK
# =========================================================


def calculate_generator_emissions(n: pypsa.Network) -> pd.DataFrame:
    """
    Approximate annual emissions by country and generator carrier in tonnes CO2.

    For generators, output is divided by efficiency to convert useful energy to
    primary fuel before multiplying by carrier CO2 intensity.
    """

    if "co2_emissions" not in n.carriers.columns:
        return pd.DataFrame(columns=["country", "carrier", "emissions_tCO2"])

    countries = country_order_from_network(n)
    records = []

    for gen in n.generators.index:
        carrier = n.generators.at[gen, "carrier"]
        if carrier not in n.carriers.index:
            continue

        co2_intensity = n.carriers.at[carrier, "co2_emissions"]
        if pd.isna(co2_intensity) or co2_intensity <= 0:
            continue

        country = get_country(gen, countries)
        if country is None:
            country = str(n.generators.at[gen, "bus"]).split("_")[0]

        efficiency = n.generators.at[gen, "efficiency"]
        if pd.isna(efficiency) or efficiency <= 0:
            efficiency = 1.0

        output = n.generators_t.p[gen].clip(lower=0)
        primary_energy = weighted_sum(n, output) / efficiency
        emissions = primary_energy * co2_intensity

        records.append({"country": country, "carrier": carrier, "emissions_tCO2": emissions})

    return pd.DataFrame(records)


def plot_task_i_emissions_by_country(n: pypsa.Network, output_dir: Path, co2_cap: float | None = None) -> None:
    """
    Plot annual CO2 emissions by country and compare total emissions with the cap.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = calculate_generator_emissions(n)

    fig, ax = plt.subplots(figsize=(9, 5))

    if df.empty:
        ax.text(0.5, 0.5, "No direct CO2 emissions", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        table = df.pivot_table(index="country", columns="carrier", values="emissions_tCO2", aggfunc="sum").fillna(0.0)
        table = table.loc[:, table.sum() > 1e-6] / 1e6
        colors = [get_color(n, c) for c in table.columns]
        table.plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_ylabel("Annual emissions [MtCO2]")
        ax.set_xlabel("")
        ax.set_title("CO2 emissions by country and technology")
        ax.grid(axis="y", alpha=0.3)
        ax.legend([nice_label(c) for c in table.columns], title="Technology", bbox_to_anchor=(1.02, 1), loc="upper left")

        total = df["emissions_tCO2"].sum() / 1e6
        cap_text = f"Cap: {co2_cap / 1e6:.2f} MtCO2" if co2_cap is not None else ""
        ax.text(
            0.02,
            0.95,
            f"Total: {total:.2f} MtCO2{cap_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    plt.tight_layout()
    plt.savefig(output_dir / "task_i_emissions_by_country.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_i_peak_heat_week_dispatch(n: pypsa.Network, output_dir: Path, focus_country: str = "DK") -> None:
    """
    Plot dispatch for the week with the highest average heat demand in one focus country.

    The optimisation remains multi-country; this is only a detailed illustration
    of one country. DK is useful if your written discussion focuses on Denmark.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)
    if focus_country not in countries:
        focus_country = countries[0]

    heat_load_name = f"{focus_country}_heat_demand"
    electricity_load_name = f"{focus_country}_electricity_demand"

    if heat_load_name in n.loads.index:
        stress_series = n.loads_t.p_set[heat_load_name]
        stress_label = f"{focus_country} heat demand"
    else:
        stress_series = n.loads_t.p_set[electricity_load_name]
        stress_label = f"{focus_country} electricity demand"

    weekly_stress = stress_series.resample("W").mean()
    peak_week_end = weekly_stress.idxmax()
    week_start = peak_week_end - pd.Timedelta(days=6)
    week_index = stress_series.loc[week_start:peak_week_end].index

    # Electricity supply in focus country
    electricity_supply = {}

    focus_gens = n.generators[n.generators.bus == focus_country]
    for gen in focus_gens.index:
        carrier = n.generators.at[gen, "carrier"]
        electricity_supply.setdefault(carrier, pd.Series(0.0, index=week_index))
        electricity_supply[carrier] += n.generators_t.p[gen].loc[week_index].clip(lower=0)

    for link in n.links.index:
        if n.links.at[link, "bus1"] != focus_country:
            continue
        carrier = n.links.at[link, "carrier"]
        if carrier in ["battery discharger", "CCGT", "H2 turbine", "H2 fuel cell"]:
            electricity_supply.setdefault(carrier, pd.Series(0.0, index=week_index))
            electricity_supply[carrier] += link_output_to_bus1(n, link).loc[week_index]

    base_electricity_load = n.loads_t.p_set[electricity_load_name].loc[week_index]

    extra_electricity_use = pd.Series(0.0, index=week_index)
    for link in n.links.index:
        if n.links.at[link, "bus0"] != focus_country:
            continue
        if n.links.at[link, "carrier"] in ["battery charger", "heat pump", "resistive heater", "electrolysis"]:
            extra_electricity_use += link_input_from_bus0(n, link).loc[week_index]

    total_electricity_use = base_electricity_load + extra_electricity_use

    # Net electricity imports to focus country
    imports = pd.Series(0.0, index=week_index)
    if not n.lines.empty:
        for line in n.lines.index:
            bus0 = n.lines.at[line, "bus0"]
            bus1 = n.lines.at[line, "bus1"]
            if focus_country not in [bus0, bus1]:
                continue
            flow = n.lines_t.p0[line].loc[week_index]
            if bus0 == focus_country:
                imports += -flow
            else:
                imports += flow

    # Heat supply
    heat_supply = {}
    heat_bus = f"{focus_country}_heat"

    for link in n.links.index:
        if n.links.at[link, "bus1"] != heat_bus:
            continue
        carrier = n.links.at[link, "carrier"]
        heat_supply[carrier] = link_output_to_bus1(n, link).loc[week_index]

    gas_boiler = f"{focus_country}_gas_boiler"
    if gas_boiler in n.generators.index:
        heat_supply["gas boiler"] = n.generators_t.p[gas_boiler].loc[week_index].clip(lower=0)

    heat_load = None
    if heat_load_name in n.loads.index:
        heat_load = n.loads_t.p_set[heat_load_name].loc[week_index]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2.5, 1.4]},
    )

    ax1, ax2, ax3 = axes

    electricity_order = [
        "solar",
        "onwind",
        "offwind",
        "coal",
        "nuclear",
        "gas CCGT",
        "CCGT",
        "battery discharger",
        "H2 turbine",
        "H2 fuel cell",
    ]
    electricity_order += [c for c in electricity_supply if c not in electricity_order]

    bottom = np.zeros(len(week_index))
    for carrier in electricity_order:
        if carrier not in electricity_supply:
            continue
        values = electricity_supply[carrier].fillna(0.0).values
        ax1.fill_between(
            week_index,
            bottom,
            bottom + values,
            label=nice_label(carrier),
            color=get_color(n, carrier),
            alpha=0.85,
        )
        bottom += values

    ax1.plot(week_index, base_electricity_load, color="black", linewidth=2.2, label="Base electricity demand")
    ax1.plot(week_index, total_electricity_use, color="black", linewidth=2.0, linestyle="--", label="Electricity demand incl. coupling")
    ax1.set_ylabel("Power [MW]")
    ax1.set_title(f"{focus_country}: electricity dispatch in peak {stress_label} week")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    heat_order = ["heat pump", "resistive heater", "gas boiler", "heat storage discharger"]
    heat_order += [c for c in heat_supply if c not in heat_order]

    bottom = np.zeros(len(week_index))
    for carrier in heat_order:
        if carrier not in heat_supply:
            continue
        values = heat_supply[carrier].fillna(0.0).values
        ax2.fill_between(
            week_index,
            bottom,
            bottom + values,
            label=nice_label(carrier),
            color=get_color(n, carrier),
            alpha=0.85,
        )
        bottom += values

    if heat_load is not None:
        ax2.plot(week_index, heat_load, color="black", linewidth=2.2, label="Heat demand")

    ax2.set_ylabel("Heat [MW]")
    ax2.set_title(f"{focus_country}: heat dispatch in the same week")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    ax3.plot(week_index, imports, color="black", linewidth=2.0, label="Net electricity imports")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_ylabel("Net imports [MW]")
    ax3.set_xlabel("Time")
    ax3.set_title(f"{focus_country}: imports help balance coupled electricity demand")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / f"task_i_{focus_country}_peak_heat_week_dispatch.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_task_i_monthly_heat_supply(n: pypsa.Network, output_dir: Path) -> None:
    """
    Monthly heat supply by technology aggregated across all countries.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = n.snapshot_weightings.generators
    countries = country_order_from_network(n)
    monthly = {}

    for country in countries:
        heat_bus = f"{country}_heat"
        for link in n.links.index:
            if n.links.at[link, "bus1"] != heat_bus:
                continue
            carrier = n.links.at[link, "carrier"]
            output = link_output_to_bus1(n, link)
            monthly.setdefault(carrier, pd.Series(0.0, index=n.snapshots))
            monthly[carrier] += output

        gas_boiler = f"{country}_gas_boiler"
        if gas_boiler in n.generators.index:
            monthly.setdefault("gas boiler", pd.Series(0.0, index=n.snapshots))
            monthly["gas boiler"] += n.generators_t.p[gas_boiler].clip(lower=0)

    if not monthly:
        print("Skipping monthly heat plot: no heat sector found.")
        return

    monthly_df = pd.DataFrame(
        {
            carrier: series.multiply(weights, axis=0).resample("ME").sum() / 1e6
            for carrier, series in monthly.items()
        }
    ).fillna(0.0)
    monthly_df = monthly_df.loc[:, monthly_df.sum() > 1e-6]

    heat_demand = pd.Series(0.0, index=n.snapshots)
    for country in countries:
        load = f"{country}_heat_demand"
        if load in n.loads.index:
            heat_demand += n.loads_t.p_set[load]
    monthly_heat_demand = heat_demand.multiply(weights, axis=0).resample("ME").sum() / 1e6

    fig, ax = plt.subplots(figsize=(11, 5))

    bottom = pd.Series(0.0, index=monthly_df.index)
    for carrier in monthly_df.columns:
        ax.bar(
            monthly_df.index,
            monthly_df[carrier],
            bottom=bottom,
            width=20,
            label=nice_label(carrier),
            color=get_color(n, carrier),
            alpha=0.85,
        )
        bottom += monthly_df[carrier]

    ax.plot(monthly_heat_demand.index, monthly_heat_demand, color="black", linewidth=2.2, label="Total heat demand")
    ax.set_ylabel("Monthly heat [TWh]")
    ax.set_title("Monthly heat supply mix across all modelled countries")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "task_i_monthly_heat_supply_all_countries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# EXPORT TABLES
# =========================================================


def export_task_i_summary_tables(n: pypsa.Network, output_dir: Path, scenario: dict) -> None:
    """
    Export CSV tables that are easy to paste into the report.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = country_order_from_network(n)

    gen_records = []
    for gen in n.generators.index:
        country = get_country(gen, countries)
        if country is None:
            continue
        gen_records.append(
            {
                "country": country,
                "technology": n.generators.at[gen, "carrier"],
                "component": gen,
                "capacity_MW": n.generators.at[gen, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else 0.0,
            }
        )

    link_records = []
    for link in n.links.index:
        country = get_country(link, countries)
        if country is None:
            continue
        link_records.append(
            {
                "country": country,
                "technology": n.links.at[link, "carrier"],
                "component": link,
                "capacity_MW": n.links.at[link, "p_nom_opt"] if "p_nom_opt" in n.links.columns else 0.0,
            }
        )

    store_records = []
    for store in n.stores.index:
        country = get_country(store, countries)
        if country is None:
            continue
        store_records.append(
            {
                "country": country,
                "technology": n.stores.at[store, "carrier"],
                "component": store,
                "energy_capacity_MWh": n.stores.at[store, "e_nom_opt"] if "e_nom_opt" in n.stores.columns else 0.0,
            }
        )

    emissions = calculate_generator_emissions(n)

    pd.DataFrame(gen_records).to_csv(output_dir / "task_i_generator_capacities_by_country.csv", index=False)
    pd.DataFrame(link_records).to_csv(output_dir / "task_i_link_capacities_by_country.csv", index=False)
    pd.DataFrame(store_records).to_csv(output_dir / "task_i_storage_capacities_by_country.csv", index=False)
    emissions.to_csv(output_dir / "task_i_emissions_by_country.csv", index=False)

    summary = {
        "objective": n.objective,
        "system_co2_cap_tCO2": scenario.get("co2_limit"),
        "calculated_direct_emissions_tCO2": float(emissions["emissions_tCO2"].sum()) if not emissions.empty else 0.0,
        "countries": ",".join(countries),
    }

    pd.Series(summary).to_csv(output_dir / "task_i_summary.csv")


def make_task_i_plots(n: pypsa.Network, output_dir: Path, scenario: dict) -> None:
    """Run the full plot package for task i."""

    plot_task_i_generation_capacity_by_country(n, output_dir)
    plot_task_i_coupling_capacity_by_country(n, output_dir)
    plot_task_i_storage_capacity_by_country(n, output_dir)
    plot_task_i_annual_electricity_generation_by_country(n, output_dir)
    plot_task_i_annual_heat_supply_by_country(n, output_dir)
    plot_task_i_electricity_use_by_country(n, output_dir)
    plot_task_i_annual_net_imports_by_country(n, output_dir)
    plot_task_i_monthly_heat_supply(n, output_dir)
    plot_task_i_emissions_by_country(n, output_dir, co2_cap=scenario.get("co2_limit"))
    plot_task_i_peak_heat_week_dispatch(n, output_dir, focus_country="DK")
    export_task_i_summary_tables(n, output_dir, scenario)


# =========================================================
# REPLACEMENT main() FOR TASK i
# =========================================================


def main() -> None:
    silence_gurobi_logger()

    base_scenario = SCENARIOS[ACTIVE_SCENARIO]
    scenario = build_task_i_interconnected_sector_coupling_scenario(base_scenario)

    print(f"Running task i scenario: {scenario['name']}")
    print(f"Weather year: {scenario['weather_year']}")
    print(f"Countries: {scenario['countries']}")
    print(f"System-wide CO2 cap: {scenario['co2_limit']:,.0f} tCO2")
    print(f"Interconnectors: {scenario['with_interconnectors']}")
    print(f"Heat sector: {scenario['with_heat_sector']}")
    print(f"CH4 network: {scenario.get('with_ch4_network', False)}")
    print(f"H2 network: {scenario.get('with_h2_network', False)}")
    print(f"Capacity caps: {scenario.get('capacity_caps', {})}")

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
        scenario=scenario,
    )

    print_model_summary(n)

    output_dir = Path("results/task_i_interconnected_sector_coupling")
    make_task_i_plots(n=n, output_dir=output_dir, scenario=scenario)

    print(f"Task i plots and summary tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
