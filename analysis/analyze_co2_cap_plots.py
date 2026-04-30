from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NETWORK_DIR = PROJECT_ROOT / "results" / "networks"
OUTPUT_DIR = PROJECT_ROOT / "results" / "co2_cap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_NETWORK_NAME = "co2_cap_baseline_2016.nc"
CAP_NETWORK_NAME = "co2_cap_cap_2016.nc"

REPORT_FONT_SIZE = 14

CARRIER_ALIASES = {
    "electricity": ["electricity", "AC"],
    "solar": ["solar"],
    "onshore wind": ["onwind", "onshore wind"],
    "offshore wind": ["offwind", "offshore wind"],
    "CCGT": ["gas CCGT", "CCGT", "gas"],
    "coal": ["coal"],
    "nuclear": ["nuclear"],
    "battery": ["battery"],
    "battery charger": ["battery charger", "battery charge"],
    "battery discharger": ["battery discharger", "battery discharge"],
    "H2 turbine": ["H2 turbine"],
    "H2 fuel cell": ["H2 fuel cell"],
    "electrolysis": ["electrolysis"],
}

DEFAULT_CARRIER_COLORS = {
    "AC": "#4C566A",
    "electricity": "#4C566A",
    "solar": "#EBCB3B",
    "onwind": "#5AA469",
    "onshore wind": "#5AA469",
    "offwind": "#2E86AB",
    "offshore wind": "#2E86AB",
    "gas CCGT": "#D08770",
    "CCGT": "#D08770",
    "coal": "#5C5C5C",
    "nuclear": "#8F6BB3",
    "battery": "#E67E22",
    "battery charger": "#C06C84",
    "battery discharger": "#6C5B7B",
    "H2 turbine": "#6FA8DC",
    "H2 fuel cell": "#5E81AC",
}

PLOT_ORDER = [
    "nuclear",
    "coal",
    "CCGT",
    "offshore wind",
    "onshore wind",
    "solar",
    "battery discharger",
    "H2 turbine",
    "H2 fuel cell",
]


# =========================================================
# STYLE
# =========================================================

def set_report_plot_style() -> None:
    """
    Set report-ready matplotlib style.
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


# =========================================================
# HELPERS
# =========================================================

def load_network(network_path: Path) -> pypsa.Network:
    """
    Load solved PyPSA network.
    """
    if not network_path.exists():
        raise FileNotFoundError(
            f"Network file not found: {network_path}\n"
            "Check the filename in NETWORK_DIR."
        )

    network = pypsa.Network(network_path)
    network.sanitize()
    return network


def map_carrier_to_display_name(carrier: str) -> str:
    """
    Map internal model carrier names to report-friendly labels.
    """
    for display_name, aliases in CARRIER_ALIASES.items():
        if carrier in aliases:
            return display_name
    return carrier


def get_carrier_color(network: pypsa.Network, carrier: str) -> str:
    """
    Return carrier color from network carrier table if available.
    Otherwise fall back to DEFAULT_CARRIER_COLORS.
    """
    display_name = map_carrier_to_display_name(carrier)

    candidates = [carrier, display_name]

    if display_name in CARRIER_ALIASES:
        candidates.extend(CARRIER_ALIASES[display_name])

    for candidate in candidates:
        if (
            candidate in network.carriers.index
            and "color" in network.carriers.columns
            and pd.notna(network.carriers.at[candidate, "color"])
            and network.carriers.at[candidate, "color"] != ""
        ):
            return network.carriers.at[candidate, "color"]

        if candidate in DEFAULT_CARRIER_COLORS:
            return DEFAULT_CARRIER_COLORS[candidate]

    return "#999999"


def get_carrier_colors(
    network: pypsa.Network,
    carriers: pd.Index | list[str],
) -> list[str]:
    """
    Return colors for multiple carriers.
    """
    return [get_carrier_color(network, carrier) for carrier in carriers]


def reorder_series(series: pd.Series, order: list[str]) -> pd.Series:
    """
    Reorder series index according to preferred order.
    """
    ordered = [carrier for carrier in order if carrier in series.index]
    remaining = [carrier for carrier in series.index if carrier not in ordered]
    return series.loc[ordered + remaining]


def get_electricity_bus_carrier(network: pypsa.Network) -> str:
    """
    Return the electricity bus carrier used in the network.
    """
    carriers = pd.Index(network.buses.carrier.dropna().unique())

    for candidate in ["AC", "electricity"]:
        if candidate in carriers:
            return candidate

    raise ValueError(
        f"Could not find electricity bus carrier. Found bus carriers: {list(carriers)}"
    )


def get_annual_electricity_mix(network: pypsa.Network) -> pd.Series:
    """
    Return annual positive electricity-bus contributions by carrier in MWh.
    """
    bus_carrier = get_electricity_bus_carrier(network)

    balance = network.statistics.energy_balance(
        aggregate_time=False,
        nice_names=False,
    )

    if "bus_carrier" not in balance.index.names:
        raise KeyError(
            "The energy balance index does not contain a 'bus_carrier' level. "
            f"Available levels are: {balance.index.names}"
        )

    balance = balance.xs(bus_carrier, level="bus_carrier")
    balance_by_carrier = balance.groupby(level="carrier").sum().T

    if len(balance_by_carrier.index) == len(network.snapshots):
        balance_by_carrier.index = network.snapshots

    balance_by_carrier = balance_by_carrier.rename(
        columns=lambda carrier: map_carrier_to_display_name(carrier)
    )

    balance_by_carrier = balance_by_carrier.T.groupby(level=0).sum().T

    annual_mix = balance_by_carrier.clip(lower=0.0).sum(axis=0)

    annual_mix = annual_mix.drop(
        labels=[
            "electricity",
            "battery",
            "battery charger",
            "electrolysis",
        ],
        errors="ignore",
    )

    annual_mix = annual_mix[annual_mix > 1e-3]
    annual_mix = reorder_series(annual_mix, PLOT_ORDER)
    annual_mix.name = "annual_generation_mwh"

    return annual_mix


def autopct_threshold(threshold: float = 3.0):
    """
    Only show percentage labels for slices larger than the threshold.
    """
    def _autopct(pct):
        return f"{pct:.1f}%" if pct >= threshold else ""
    return _autopct


# =========================================================
# PLOTTING
# =========================================================

def plot_single_pie(
    network: pypsa.Network,
    annual_mix: pd.Series,
    output_path: Path,
) -> None:
    """
    Plot one clean donut chart for annual electricity generation mix.
    """
    colors = get_carrier_colors(network, annual_mix.index)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    wedges, _, _ = ax.pie(
        annual_mix,
        colors=colors,
        labels=None,
        autopct=autopct_threshold(3.0),
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        radius=0.92,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1.5,
            "width": 0.88,
        },
        textprops={
            "color": "black",
            "fontsize": REPORT_FONT_SIZE,
        },
    )

    ax.axis("equal")

    ax.legend(
        wedges,
        annual_mix.index,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.12),
        frameon=True,
        framealpha=0.9,
        facecolor="white",
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_combined_pies(
    base_network: pypsa.Network,
    base_mix: pd.Series,
    cap_network: pypsa.Network,
    cap_mix: pd.Series,
    output_path: Path,
) -> None:
    """
    Plot baseline and CO2 cap annual electricity mix side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.patch.set_facecolor("white")

    cases = [
        (axes[0], base_network, base_mix),
        (axes[1], cap_network, cap_mix),
    ]

    for ax, network, annual_mix in cases:
        colors = get_carrier_colors(network, annual_mix.index)

        wedges, _, _ = ax.pie(
            annual_mix,
            colors=colors,
            labels=None,
            autopct=autopct_threshold(3.0),
            startangle=90,
            counterclock=False,
            pctdistance=0.72,
            radius=0.90,
            wedgeprops={
                "edgecolor": "white",
                "linewidth": 1.5,
                "width": 0.88,
            },
            textprops={
                "color": "black",
                "fontsize": REPORT_FONT_SIZE - 1,
            },
        )

        ax.axis("equal")

        ax.legend(
            wedges,
            annual_mix.index,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.10),
            frameon=True,
            framealpha=0.9,
            facecolor="white",
            fontsize=REPORT_FONT_SIZE - 1,
        )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_comparison_bar(
    base_mix: pd.Series,
    cap_mix: pd.Series,
    output_path: Path,
) -> None:
    """
    Optional comparison plot: horizontal grouped bar chart.
    """
    all_carriers = list(dict.fromkeys(list(base_mix.index) + list(cap_mix.index)))

    comparison = pd.DataFrame(
        {
            "Base case": base_mix.reindex(all_carriers).fillna(0.0) / 1000.0,
            "CO$_2$ cap case": cap_mix.reindex(all_carriers).fillna(0.0) / 1000.0,
        }
    )

    comparison = comparison.loc[
        [carrier for carrier in PLOT_ORDER if carrier in comparison.index]
        + [carrier for carrier in comparison.index if carrier not in PLOT_ORDER]
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    comparison.plot.barh(
        ax=ax,
        width=0.75,
    )

    ax.set_xlabel("Annual generation [GWh]")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    set_report_plot_style()

    base_network_path = NETWORK_DIR / BASE_NETWORK_NAME
    cap_network_path = NETWORK_DIR / CAP_NETWORK_NAME

    base_network = load_network(base_network_path)
    cap_network = load_network(cap_network_path)

    base_mix = get_annual_electricity_mix(base_network)
    cap_mix = get_annual_electricity_mix(cap_network)

    base_mix.to_csv(OUTPUT_DIR / "annual_electricity_mix_base_case.csv")
    cap_mix.to_csv(OUTPUT_DIR / "annual_electricity_mix_co2_cap_case.csv")

    print("\nAnnual electricity mix, base case [GWh]:")
    print((base_mix / 1000.0).round(2))

    print("\nAnnual electricity mix, CO2 cap case [GWh]:")
    print((cap_mix / 1000.0).round(2))

    plot_single_pie(
        network=base_network,
        annual_mix=base_mix,
        output_path=OUTPUT_DIR / "annual_electricity_mix_base_case.png",
    )

    plot_single_pie(
        network=cap_network,
        annual_mix=cap_mix,
        output_path=OUTPUT_DIR / "annual_electricity_mix_co2_cap_case.png",
    )

    plot_combined_pies(
        base_network=base_network,
        base_mix=base_mix,
        cap_network=cap_network,
        cap_mix=cap_mix,
        output_path=OUTPUT_DIR / "annual_electricity_mix_base_vs_co2_cap.png",
    )

    plot_comparison_bar(
        base_mix=base_mix,
        cap_mix=cap_mix,
        output_path=OUTPUT_DIR / "annual_electricity_mix_base_vs_co2_cap_bar.png",
    )

    print(f"\nPlots and CSV files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()