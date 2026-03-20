from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from model import prepare_costs, load_dk_timeseries, create_network_dk


OUTPUT_DIR = Path("results/interannual_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_yearly_optims(
    years,
    cost_file,
    timeseries_file,
    financial_parameters,
    solver_name="gurobi",
):
    results = []

    cost_data = prepare_costs(
        cost_file=cost_file,
        financial_parameters=financial_parameters,
        number_of_years=financial_parameters["nyears"],
    )

    for year in years:
        print(f"Running optimization for weather year {year}...")
        timeseries_data = load_dk_timeseries(
            timeseries_file=timeseries_file,
            year=str(year),
        )

        n = create_network_dk(
            cost_data=cost_data,
            timeseries_data=timeseries_data,
            co2_price=financial_parameters["co2_price"],
        )

        n.optimize(solver_name=solver_name)

        gen_capacity = n.generators.p_nom_opt.copy()
        if gen_capacity is None:
            gen_capacity = pd.Series(dtype=float)
        dispatch = n.generators_t.p if hasattr(n.generators_t, "p") else pd.DataFrame()

        # If dispatch has MultiIndex columns, flatten to names.
        if isinstance(dispatch.columns, pd.MultiIndex):
            dispatch.columns = [" ".join(map(str, c)).strip() for c in dispatch.columns]

        # Ensure we have a row per generator for capacity/variability
        generator_names = list(n.generators.index)
        capacity_values = gen_capacity.reindex(generator_names).fillna(0.0).values

        if dispatch.empty:
            mean_values = [0.0] * len(generator_names)
            std_values = [0.0] * len(generator_names)
        else:
            mean_values = dispatch.mean(axis=0).reindex(generator_names).fillna(0.0).values
            std_values = dispatch.std(axis=0).reindex(generator_names).fillna(0.0).values

        capacity_factor_values = []
        for cap, mean_d in zip(capacity_values, mean_values):
            if cap > 0:
                capacity_factor_values.append(mean_d / cap)
            else:
                capacity_factor_values.append(0.0)

        year_df = pd.DataFrame(
            {
                "generator": generator_names,
                "capacity_mw": capacity_values,
                "mean_dispatch_mw": mean_values,
                "dispatch_std_mw": std_values,
                "capacity_factor": capacity_factor_values,
                "year": int(year),
            }
        )
        results.append(year_df)

    combined = pd.concat(results, ignore_index=True).sort_values(["generator", "year"])
    combined.to_csv(OUTPUT_DIR / "interannual_capacity_variability.csv", index=False)
    return combined


def plot_sensitivity(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.copy()
    generators = grouped["generator"].unique()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    for year in sorted(grouped["year"].unique()):
        year_df = grouped[grouped["year"] == year].set_index("generator")
        axes[0].plot(
            generators,
            [year_df.loc[g, "capacity_mw"] if g in year_df.index else 0 for g in generators],
            marker="o",
            label=str(year),
        )
        axes[1].plot(
            generators,
            [year_df.loc[g, "dispatch_std_mw"] if g in year_df.index else 0 for g in generators],
            marker="o",
            label=str(year),
        )

    axes[0].set_title("Interannual sensitivity: Optimal capacity by generator")
    axes[0].set_ylabel("Capacity [MW]")
    axes[0].set_xticks(range(len(generators)))
    axes[0].set_xticklabels(generators, rotation=45, ha="right")
    axes[0].legend(title="Weather year")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Interannual sensitivity: Dispatch variability by generator")
    axes[1].set_ylabel("Dispatch std dev [MW]")
    axes[1].set_xticks(range(len(generators)))
    axes[1].set_xticklabels(generators, rotation=45, ha="right")
    axes[1].legend(title="Weather year")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        ax.set_xlabel("Generator")

    fig.savefig(output_dir / "interannual_capacity_variability.png", dpi=200)
    plt.close(fig)

    # Also make more direct bar plots
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    for year in sorted(grouped["year"].unique()):
        subset = grouped[grouped["year"] == year]
        ax2.bar(
            subset["generator"].astype(str) + " " + subset["year"].astype(str),
            subset["capacity_mw"],
            alpha=0.7,
            label=str(year),
        )
    ax2.set_title("Optimal capacity by generator and weather year")
    ax2.set_ylabel("Capacity [MW]")
    ax2.set_xticklabels([], rotation=45)
    ax2.grid(alpha=0.2)
    fig2.autofmt_xdate(rotation=90)
    fig2.savefig(output_dir / "interannual_capacity_bar.png", dpi=200)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    for year in sorted(grouped["year"].unique()):
        subset = grouped[grouped["year"] == year]
        ax3.bar(
            subset["generator"].astype(str) + " " + subset["year"].astype(str),
            subset["dispatch_std_mw"],
            alpha=0.7,
            label=str(year),
        )
    ax3.set_title("Dispatch variability by generator and weather year")
    ax3.set_ylabel("Dispatch std dev [MW]")
    ax3.set_xticklabels([], rotation=90)
    ax3.grid(alpha=0.2)
    fig3.autofmt_xdate(rotation=90)
    fig3.savefig(output_dir / "interannual_dispatch_variability_bar.png", dpi=200)
    plt.close(fig3)

    # Plot average capacity factor by generator and year
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 5))
    for year in sorted(grouped["year"].unique()):
        subset = grouped[grouped["year"] == year]
        ax4.plot(
            subset["generator"],
            subset["capacity_factor"],
            marker="o",
            label=str(year),
        )
    ax4.set_title("Average capacity factor by generator")
    ax4.set_ylabel("Capacity factor [-]")
    ax4.set_xticks(range(len(generators)))
    ax4.set_xticklabels(generators, rotation=45, ha="right")
    ax4.legend(title="Year")
    ax4.grid(alpha=0.2)
    fig4.tight_layout()
    fig4.savefig(output_dir / "interannual_capacity_factor_by_generator.png", dpi=200)
    plt.close(fig4)

    # Plot real capacity (mean dispatch -> effective utilization) by generator and year
    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 5))
    for year in sorted(grouped["year"].unique()):
        subset = grouped[grouped["year"] == year]
        ax5.plot(
            subset["generator"],
            subset["mean_dispatch_mw"],
            marker="o",
            label=str(year),
        )
    ax5.set_title("Average generated (real capacity) by generator")
    ax5.set_ylabel("Mean dispatch [MW]")
    ax5.set_xticks(range(len(generators)))
    ax5.set_xticklabels(generators, rotation=45, ha="right")
    ax5.legend(title="Year")
    ax5.grid(alpha=0.2)
    fig5.tight_layout()
    fig5.savefig(output_dir / "interannual_mean_dispatch_by_generator.png", dpi=200)
    plt.close(fig5)


def plot_capacity_summary(df: pd.DataFrame, output_dir: Path) -> None:
    df["year"] = df["year"].astype(int)
    pivot = df.pivot(index="generator", columns="year", values="capacity_mw")
    pivot.columns = pivot.columns.astype(int)
    pivot = pivot.reindex(columns=[2015, 2016, 2017])

    stats = pivot.rename(columns={2015: "2015", 2016: "2016", 2017: "2017"}).copy()
    stats["mean"] = stats[["2015", "2016", "2017"]].mean(axis=1)
    stats["min"] = stats[["2015", "2016", "2017"]].min(axis=1)
    stats["max"] = stats[["2015", "2016", "2017"]].max(axis=1)

    stats.to_csv(output_dir / "part_b_capacity_stats.csv")

    print("Installed capacities by generator and year with summary:")
    print(stats)

    # Plot 1: grouped bar chart by year
    generators = stats.index.tolist()
    x = range(len(generators))
    width = 0.25

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar([xi - width for xi in x], stats["2015"], width=width, label="2015")
    ax1.bar(x, stats["2016"], width=width, label="2016")
    ax1.bar([xi + width for xi in x], stats["2017"], width=width, label="2017")
    ax1.set_title("Optimized installed capacity by generator and weather year")
    ax1.set_xlabel("Generator")
    ax1.set_ylabel("Installed capacity [MW]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(generators, rotation=45, ha="right")
    ax1.legend(title="Weather year")
    ax1.grid(axis="y", alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(output_dir / "part_b_grouped_capacity_by_generator_year.png", dpi=200)
    plt.close(fig1)

    # Plot 2: mean with min-max whiskers
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(x, stats["mean"], yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]], capsize=5)
    ax2.set_title("Average installed capacity and interannual variability by generator")
    ax2.set_xlabel("Generator")
    ax2.set_ylabel("Mean installed capacity [MW]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(generators, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(output_dir / "part_b_mean_variability_capacity_by_generator.png", dpi=200)
    plt.close(fig2)

    # Alternative variability plot: line with shaded min-max range
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(x, stats["mean"], marker="o", linestyle="-", color="navy", label="Mean")
    ax3.fill_between(x, stats["min"], stats["max"], color="skyblue", alpha=0.3, label="Min-Max range")
    ax3.set_title("Average installed capacity with min-max variability")
    ax3.set_xlabel("Generator")
    ax3.set_ylabel("Installed capacity [MW]")
    ax3.set_xticks(x)
    ax3.set_xticklabels(generators, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.2)
    fig3.tight_layout()
    fig3.savefig(output_dir / "part_b_shaded_min_max_capacity.png", dpi=200)
    plt.close(fig3)


def main() -> None:
    years = [2015, 2016, 2017]
    financial_parameters = {
        "fill_values": 0.0,
        "r": 0.07,
        "nyears": 1,
        "year": 2025,
        "co2_price": 80.0,
    }

    cost_file = f"cost_data/costs_{financial_parameters['year']}.csv"
    timeseries_file = "Data/time_series_60min_singleindex_filtered_DK.csv"

    df = run_yearly_optims(
        years=years,
        cost_file=cost_file,
        timeseries_file=timeseries_file,
        financial_parameters=financial_parameters,
        solver_name="gurobi",
    )
    print(df.head())
    plot_sensitivity(df, OUTPUT_DIR)
    plot_capacity_summary(df, OUTPUT_DIR)
    print(f"Saved interannual results and plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
