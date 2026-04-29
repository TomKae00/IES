import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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