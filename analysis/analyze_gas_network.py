import csv
import pathlib
import matplotlib.pyplot as plt
import pypsa

# Approximate geographic centres used for the network map (lon, lat).
COUNTRY_POS = {
    "DK": (10.5, 56.0),
    "DE": (10.0, 51.5),
    "SE": (16.0, 62.0),
    "NO": (9.0,  64.5),
}

CO2_PRICES = [0, 50, 80, 100, 150, 200, 300] # for sensitivity analysis — these should match the prices used in run_model.py


def load_network(path):
    return pypsa.Network(path)


def create_output_folder():
    folder = pathlib.Path("results/gas_network_analysis")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# Network map
# ---------------------------------------------------------------------------

def _corridor_flows(n, carrier):
    """Return {(country_a, country_b): total_flow_TWh} per unique corridor."""
    pipes = n.links[n.links.carrier == carrier]
    flows = {}
    for link in pipes.index:
        a = n.links.loc[link, "bus0"].split("_")[0]
        b = n.links.loc[link, "bus1"].split("_")[0]
        key = tuple(sorted([a, b]))
        flows[key] = flows.get(key, 0.0) + n.links_t.p0[link].clip(lower=0).sum() / 1e6
    return flows


def _draw_network_map(ax, flows, color, title):
    """Draw one network map panel. Line thickness is normalised within the panel."""
    max_flow = max(flows.values()) if flows else 1.0

    for (a, b), flow in flows.items():
        xa, ya = COUNTRY_POS[a]
        xb, yb = COUNTRY_POS[b]
        lw = 0.5 + 9.0 * (flow / max_flow)
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw,
                solid_capstyle="round", alpha=0.8, zorder=2)
        if flow > 0.01:
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            ax.text(mx, my, f"{flow:.1f} TWh",
                    ha="center", va="center", fontsize=15, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="none", alpha=0.9))

    for country, (x, y) in COUNTRY_POS.items():
        ax.scatter(x, y, s=1200, color="white", edgecolors="black",
                   linewidths=2.5, zorder=5)
        ax.text(x, y, country, ha="center", va="center",
                fontweight="bold", fontsize=17, zorder=6)

    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_xlim(4, 22)
    ax.set_ylim(48, 69)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2)


def plot_network_maps(n_ch4, n_h2, folder):
    """Two side-by-side maps showing pipeline flows for CH4 and H2 scenarios."""
    ch4_flows = _corridor_flows(n_ch4, "CH4 pipeline")
    h2_flows  = _corridor_flows(n_h2,  "H2 pipeline")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    _draw_network_map(ax1, ch4_flows, "#D08770", "CH₄ pipeline network (2019)")
    _draw_network_map(ax2, h2_flows,  "#5E81AC", "H₂ pipeline network (2019)")

    plt.tight_layout()
    plt.savefig(folder / "network_maps.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved network_maps.png")


# ---------------------------------------------------------------------------
# CO2 sensitivity
# ---------------------------------------------------------------------------

def _pipeline_twh(n, carrier):
    """Total annual pipeline throughput in TWh for a given carrier."""
    pipes = n.links[n.links.carrier == carrier]
    if pipes.empty:
        return 0.0
    return n.links_t.p0[pipes.index].clip(lower=0).sum().sum() / 1e6


def _find_sensitivity_file(carrier, price):
    """Return the Path of a pre-solved sensitivity network, or None if missing.

    Tries the canonical naming produced by run_model.py first, then falls back
    to the main scenario networks for the CO₂ prices they were solved at.
    """
    base = pathlib.Path("results/networks")
    if carrier == "ch4":
        candidates = [base / f"sensitivity_gas_ch4_only_co2_{price}_2019.nc"]
        if price == 80:
            candidates.append(base / "gas_ch4_only_2019.nc")
    else:
        candidates = [base / f"sensitivity_gas_H2_only_co2_{price}_2019.nc"]
        if price == 80:
            candidates.append(base / "gas_H2_only_co2_80_2019.nc")
        if price == 200:
            candidates.append(base / "gas_H2_only_2019.nc")

    return next((p for p in candidates if p.exists()), None)


def _load_sensitivity_data():
    """Load available sensitivity networks and return throughput per CO₂ price.

    Returns (ch4_prices, ch4_vals, h2_prices, h2_vals).
    Prints a warning for each missing file.
    """
    ch4_prices, ch4_vals = [], []
    h2_prices,  h2_vals  = [], []

    for price in CO2_PRICES:
        ch4_path = _find_sensitivity_file("ch4", price)
        h2_path  = _find_sensitivity_file("h2",  price)

        if ch4_path:
            ch4_prices.append(price)
            ch4_vals.append(_pipeline_twh(load_network(ch4_path), "CH4 pipeline"))
        else:
            print(f"  Missing CH4 co2={price} — run 'sensitivity_gas_ch4_only_co2_{price}' in run_model.py")

        if h2_path:
            h2_prices.append(price)
            h2_vals.append(_pipeline_twh(load_network(h2_path), "H2 pipeline"))
        else:
            print(f"  Missing H2  co2={price} — run 'sensitivity_gas_H2_only_co2_{price}' in run_model.py")

    return ch4_prices, ch4_vals, h2_prices, h2_vals


def plot_co2_sensitivity(folder):
    """Pipeline throughput vs CO₂ price — dual y-axis version."""
    ch4_prices, ch4_vals, h2_prices, h2_vals = _load_sensitivity_data()

    if not ch4_prices and not h2_prices:
        print("No sensitivity networks found. Skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(ch4_prices, ch4_vals, "o-", color="#D08770", linewidth=2,
                   markersize=6, label="CH₄ pipeline (left axis)")
    l2, = ax2.plot(h2_prices,  h2_vals,  "s--", color="#5E81AC", linewidth=2,
                   markersize=6, label="H₂ pipeline (right axis)")

    ax1.set_xlabel("CO₂ price [EUR/tCO₂]")
    ax1.set_ylabel("CH₄ pipeline throughput [TWh]", color="#D08770")
    ax2.set_ylabel("H₂ pipeline throughput [TWh]",  color="#5E81AC")
    ax1.tick_params(axis="y", labelcolor="#D08770")
    ax2.tick_params(axis="y", labelcolor="#5E81AC")
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.legend(handles=[l1, l2], loc="upper left")
    ax1.set_title("Gas pipeline throughput vs CO₂ price (2019)")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(folder / "co2_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved co2_sensitivity.png")


def plot_co2_sensitivity_sidebyside(folder):
    """Pipeline throughput vs CO₂ price — two separate panels, one per carrier."""
    ch4_prices, ch4_vals, h2_prices, h2_vals = _load_sensitivity_data()

    if not ch4_prices and not h2_prices:
        print("No sensitivity networks found. Skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ch4_prices, ch4_vals, "o-", color="#D08770", linewidth=2, markersize=6)
    ax1.set_title("CH₄ pipeline throughput vs CO₂ price")
    ax1.set_xlabel("CO₂ price [EUR/tCO₂]")
    ax1.set_ylabel("Annual throughput [TWh]")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2.plot(h2_prices, h2_vals, "s-", color="#5E81AC", linewidth=2, markersize=6)
    ax2.set_title("H₂ pipeline throughput vs CO₂ price")
    ax2.set_xlabel("CO₂ price [EUR/tCO₂]")
    ax2.set_ylabel("Annual throughput [TWh]")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Gas pipeline throughput sensitivity to CO₂ price (2019)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(folder / "co2_sensitivity_sidebyside.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved co2_sensitivity_sidebyside.png")


# ---------------------------------------------------------------------------
# Electricity vs gas comparison
# ---------------------------------------------------------------------------

def _elec_corridor_flows(n):
    """Return {(country_a, country_b): bidirectional_flow_TWh} for AC lines."""
    flows = {}
    for line in n.lines.index:
        a = n.lines.loc[line, "bus0"].split("_")[0]
        b = n.lines.loc[line, "bus1"].split("_")[0]
        key = tuple(sorted([a, b]))
        flows[key] = flows.get(key, 0.0) + n.lines_t.p0[line].abs().sum() / 1e6
    return flows


def compare_electricity_vs_gas(n_ch4, n_h2, folder):
    """Print a corridor-level comparison of electricity vs gas pipeline throughput.

    For electricity, checks whether n_ch4 already contains AC lines; if not,
    falls back to loading interconnected_2016.nc. Results are printed to stdout
    and saved to electricity_vs_gas_comparison.csv.
    """
    if len(n_ch4.lines) > 0:
        n_elec = n_ch4
        elec_label = "CH4-scenario AC lines (2019)"
    else:
        elec_path = pathlib.Path("results/networks/interconnected_2016.nc")
        if not elec_path.exists():
            print("No electricity network found — skipping electricity vs gas comparison.")
            return
        n_elec = load_network(elec_path)
        elec_label = "interconnected_2016 (different year from gas scenarios — cross-year indicative)"

    elec_flows = _elec_corridor_flows(n_elec)
    ch4_flows  = _corridor_flows(n_ch4, "CH4 pipeline")
    h2_flows   = _corridor_flows(n_h2,  "H2 pipeline")

    all_corridors = sorted(set(elec_flows) | set(ch4_flows) | set(h2_flows))

    total_elec = sum(elec_flows.values())
    total_ch4  = sum(ch4_flows.values())
    total_h2   = sum(h2_flows.values())

    print("\n" + "=" * 68)
    print("ELECTRICITY vs GAS NETWORK COMPARISON")
    print(f"Electricity source: {elec_label}")
    print("=" * 68)

    print(f"\n{'Corridor':<10} {'Electricity [TWh]':>20} {'CH4 [TWh]':>12} {'H2 [TWh]':>11} {'Dominant':>12}")
    print("-" * 67)

    rows = []
    for a, b in all_corridors:
        elec = elec_flows.get((a, b), 0.0)
        ch4  = ch4_flows.get((a, b), 0.0)
        h2   = h2_flows.get((a, b), 0.0)
        dominant = max([("electricity", elec), ("CH4", ch4), ("H2", h2)], key=lambda x: x[1])[0]
        print(f"{a}-{b:<7} {elec:>20.1f} {ch4:>12.1f} {h2:>11.1f} {dominant:>12}")
        rows.append({"corridor": f"{a}-{b}", "electricity_TWh": round(elec, 2),
                     "CH4_TWh": round(ch4, 2), "H2_TWh": round(h2, 2), "dominant": dominant})

    print("-" * 67)
    print(f"{'TOTAL':<10} {total_elec:>20.1f} {total_ch4:>12.1f} {total_h2:>11.1f}")

    print("\n--- System-level ratios ---")
    if total_ch4 > 0:
        print(f"  Electricity / CH4 : {total_elec / total_ch4:.2f}x")
    if total_h2 > 0:
        print(f"  Electricity / H2  : {total_elec / total_h2:.2f}x")
    if total_ch4 > 0 and total_h2 > 0:
        print(f"  CH4 / H2          : {total_ch4 / total_h2:.2f}x")

    print("\n--- Corridors where gas throughput exceeds electricity ---")
    any_gas_dominant = False
    for a, b in all_corridors:
        elec = elec_flows.get((a, b), 0.0)
        ch4  = ch4_flows.get((a, b), 0.0)
        h2   = h2_flows.get((a, b), 0.0)
        gas_max = max(ch4, h2)
        if gas_max > elec:
            any_gas_dominant = True
            gas_type = "CH4" if ch4 >= h2 else "H2"
            print(f"  {a}-{b}: {gas_type} pipeline ({gas_max:.1f} TWh) > electricity ({elec:.1f} TWh), ratio {gas_max / elec:.2f}x")
    if not any_gas_dominant:
        print("  Electricity dominates on every corridor.")

    csv_path = folder / "electricity_vs_gas_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["corridor", "electricity_TWh", "CH4_TWh", "H2_TWh", "dominant"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved electricity_vs_gas_comparison.csv")
    print("=" * 68 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_ch4 = load_network("results/networks/gas_ch4_only_2019.nc")
    n_h2  = load_network("results/networks/gas_H2_only_2019.nc")
    folder = create_output_folder()

    plot_network_maps(n_ch4, n_h2, folder)
    plot_co2_sensitivity(folder)
    plot_co2_sensitivity_sidebyside(folder)
    compare_electricity_vs_gas(n_ch4, n_h2, folder)

    print(f"Done. Results in {folder}")


if __name__ == "__main__":
    main()
