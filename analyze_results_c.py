from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa

MODEL_PATH_WITH_BATTERY = "/home/tom/PycharmProjects/IES/results/dk_base_battery_network_2016.nc"
MODEL_PATH_NO_BATTERY = "/home/tom/PycharmProjects/IES/results/dk_base_network_2016.nc"
OUTPUT_DIR = Path("/home/tom/PycharmProjects/IES/results/storage_analysis")

CARRIER_ORDER = [
    "solar",
    "offwind",
    "gas",
    "battery discharger",
    "battery charger",
    "electricity",
]


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_network(model_path: str) -> pypsa.Network:
    """
    Load solved PyPSA network.
    """
    return pypsa.Network(model_path)


def reorder_carriers(df: pd.DataFrame, carrier_order: list[str]) -> pd.DataFrame:
    """
    Reorder carriers for plotting.
    """
    ordered_existing = [carrier for carrier in carrier_order if carrier in df.columns]
    remaining = [carrier for carrier in df.columns if carrier not in ordered_existing]
    return df[ordered_existing + remaining]


def drop_empty_carriers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove carriers that are all NaN or exactly zero.
    """
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, (df.fillna(0).abs() > 0).any(axis=0)]
    return df


def get_energy_balance_by_carrier(n: pypsa.Network) -> pd.DataFrame:
    """
    Return the time-resolved energy balance at the electricity bus,
    aggregated by carrier.
    """
    balance = n.statistics.energy_balance(aggregate_time=False)
    balance_dk = balance.xs("electricity", level="bus_carrier")
    balance_by_carrier = balance_dk.groupby(level="carrier").sum()
    balance_by_carrier_t = balance_by_carrier.T
    balance_by_carrier_t.index = n.snapshots
    balance_by_carrier_t = drop_empty_carriers(balance_by_carrier_t)
    balance_by_carrier_t = reorder_carriers(balance_by_carrier_t, CARRIER_ORDER)
    return balance_by_carrier_t


def get_carrier_colors(n: pypsa.Network, columns: pd.Index) -> list[str]:
    """
    Return colors for carriers based on n.carriers['color'].
    """
    fallback_color = "#999999"
    colors = []

    for carrier in columns:
        if carrier in n.carriers.index and pd.notna(n.carriers.at[carrier, "color"]):
            colors.append(n.carriers.at[carrier, "color"])
        else:
            colors.append(fallback_color)

    return colors


def plot_full_year_energy_balance(
    n: pypsa.Network,
    balance_by_carrier_t: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot full-year system energy balance by carrier.
    """
    plot_data = drop_empty_carriers(balance_by_carrier_t.copy())
    plot_data = reorder_carriers(plot_data, CARRIER_ORDER)
    colors = get_carrier_colors(n, plot_data.columns)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    plot_data.plot.area(stacked=True, ax=ax, color=colors)

    ax.set_ylabel("Power balance [MW]", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10, ncol=2)

    plt.tight_layout()
    fig.savefig(output_dir / "energy_balance_full_year.png", bbox_inches="tight")
    plt.close(fig)


def plot_energy_balance_week(
    n: pypsa.Network,
    balance_by_carrier_t: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_path: Path,
) -> None:
    """
    Plot one representative week of the system energy balance.
    """
    week_balance = balance_by_carrier_t.loc[start_date:end_date].copy()
    week_balance = drop_empty_carriers(week_balance)
    week_balance = reorder_carriers(week_balance, CARRIER_ORDER)
    colors = get_carrier_colors(n, week_balance.columns)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    week_balance.plot.area(stacked=True, ax=ax, color=colors)

    ax.set_ylabel("Power balance [MW]", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=5, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_battery_soc(n: pypsa.Network, output_dir: Path) -> None:
    """
    Plot the full-year battery state of charge.
    """
    soc = n.stores_t.e["DK_battery_store"]

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(soc.index, soc, linewidth=1.2)

    ax.set_ylabel("State of charge [MWh]", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "battery_soc_full_year.png", bbox_inches="tight")
    plt.close(fig)


def get_battery_summary(n: pypsa.Network) -> pd.Series:
    """
    Calculate key battery metrics.
    """
    energy_capacity = n.stores.at["DK_battery_store", "e_nom_opt"]
    charge_capacity = n.links.at["DK_battery_charger", "p_nom_opt"]
    discharge_capacity = n.links.at["DK_battery_discharger", "p_nom_opt"]

    soc = n.stores_t.e["DK_battery_store"]
    charge_ts = n.links_t.p0["DK_battery_charger"].clip(lower=0)
    discharge_ts = n.links_t.p1["DK_battery_discharger"].abs()

    annual_charge = charge_ts.sum()
    annual_discharge = discharge_ts.sum()
    duration_hours = energy_capacity / discharge_capacity if discharge_capacity > 0 else float("nan")
    equivalent_cycles = annual_discharge / energy_capacity if energy_capacity > 0 else float("nan")

    summary = pd.Series(
        {
            "Battery energy capacity [MWh]": energy_capacity,
            "Battery charging power [MW]": charge_capacity,
            "Battery discharging power [MW]": discharge_capacity,
            "Storage duration [h]": duration_hours,
            "Annual charged energy [MWh]": annual_charge,
            "Annual discharged energy [MWh]": annual_discharge,
            "Equivalent full cycles [-]": equivalent_cycles,
            "Average SOC [MWh]": soc.mean(),
            "Maximum SOC [MWh]": soc.max(),
            "Minimum SOC [MWh]": soc.min(),
            "Hours at empty SOC [-]": (soc <= 1e-3).sum(),
            "Hours at full SOC [-]": (soc >= 0.99 * energy_capacity).sum(),
        }
    )

    return summary


def print_battery_summary(n: pypsa.Network) -> None:
    """
    Print key battery metrics.
    """
    summary = get_battery_summary(n)

    print("\nBATTERY SUMMARY")
    print("-" * 60)
    for label, value in summary.items():
        print(f"{label:<35} {value:>12.2f}")


def get_generation_by_carrier(n: pypsa.Network) -> pd.Series:
    """
    Return annual electricity generation by carrier [MWh].
    """
    generation = {}

    if not n.generators.empty:
        gen_energy = n.generators_t.p.sum(axis=0)
        gen_by_carrier = gen_energy.groupby(n.generators.carrier).sum()
        for carrier, value in gen_by_carrier.items():
            generation[carrier] = generation.get(carrier, 0.0) + value

    if not n.links.empty:
        link_energy = n.links_t.p1.sum(axis=0).abs()
        link_by_carrier = link_energy.groupby(n.links.carrier).sum()
        for carrier, value in link_by_carrier.items():
            generation[carrier] = generation.get(carrier, 0.0) + value

    return pd.Series(generation).sort_index()


def get_installed_capacities_by_carrier(n: pypsa.Network) -> pd.Series:
    """
    Return installed capacities by carrier [MW or MWh depending on asset].
    """
    capacities = {}

    if not n.generators.empty:
        if "p_nom_opt" in n.generators.columns:
            gen_caps = n.generators.groupby("carrier")["p_nom_opt"].sum()
        else:
            gen_caps = n.generators.groupby("carrier")["p_nom"].sum()

        for carrier, value in gen_caps.items():
            capacities[f"{carrier} generation capacity [MW]"] = value

    if not n.links.empty:
        if "p_nom_opt" in n.links.columns:
            link_caps = n.links.groupby("carrier")["p_nom_opt"].sum()
        else:
            link_caps = n.links.groupby("carrier")["p_nom"].sum()

        for carrier, value in link_caps.items():
            capacities[f"{carrier} link capacity [MW]"] = value

    if not n.stores.empty:
        if "e_nom_opt" in n.stores.columns:
            store_caps = n.stores.groupby("carrier")["e_nom_opt"].sum()
        else:
            store_caps = n.stores.groupby("carrier")["e_nom"].sum()

        for carrier, value in store_caps.items():
            capacities[f"{carrier} energy capacity [MWh]"] = value

    return pd.Series(capacities).sort_index()


def compare_series(
    without_battery: pd.Series,
    with_battery: pd.Series,
    value_name: str,
) -> pd.DataFrame:
    """
    Compare a metric with and without battery.
    """
    df = pd.concat(
        [without_battery.rename("without_battery"), with_battery.rename("with_battery")],
        axis=1,
    ).fillna(0.0)

    df["absolute_change"] = df["with_battery"] - df["without_battery"]
    df["relative_change_%"] = df["absolute_change"] / df["without_battery"].replace(0, pd.NA) * 100
    df.index.name = value_name
    return df.sort_index()


def print_generation_comparison(n_without: pypsa.Network, n_with: pypsa.Network) -> None:
    """
    Print annual generation comparison by carrier.
    """
    gen_without = get_generation_by_carrier(n_without)
    gen_with = get_generation_by_carrier(n_with)
    comparison = compare_series(gen_without, gen_with, "carrier")

    print("\nANNUAL GENERATION BY CARRIER [MWh]")
    print("-" * 60)
    print(comparison.round(2).to_string())


def print_capacity_comparison(n_without: pypsa.Network, n_with: pypsa.Network) -> None:
    """
    Print installed capacity comparison.
    """
    cap_without = get_installed_capacities_by_carrier(n_without)
    cap_with = get_installed_capacities_by_carrier(n_with)
    comparison = compare_series(cap_without, cap_with, "asset")

    print("\nINSTALLED CAPACITY COMPARISON")
    print("-" * 60)
    print(comparison.round(2).to_string())


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    n_with_battery = load_network(MODEL_PATH_WITH_BATTERY)
    n_without_battery = load_network(MODEL_PATH_NO_BATTERY)

    balance_by_carrier_t = get_energy_balance_by_carrier(n_with_battery)

    print_battery_summary(n_with_battery)
    print_generation_comparison(n_without_battery, n_with_battery)
    print_capacity_comparison(n_without_battery, n_with_battery)

    plot_full_year_energy_balance(n_with_battery, balance_by_carrier_t, OUTPUT_DIR)

    plot_energy_balance_week(
        n=n_with_battery,
        balance_by_carrier_t=balance_by_carrier_t,
        start_date="2016-06-13",
        end_date="2016-06-19 23:00:00",
        output_path=OUTPUT_DIR / "energy_balance_summer_week.png",
    )

    plot_energy_balance_week(
        n=n_with_battery,
        balance_by_carrier_t=balance_by_carrier_t,
        start_date="2016-01-11",
        end_date="2016-01-17 23:00:00",
        output_path=OUTPUT_DIR / "energy_balance_winter_week.png",
    )

    plot_battery_soc(n_with_battery, OUTPUT_DIR)

    print("\nSaved plots to:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(file)


if __name__ == "__main__":
    main()