import pandas as pd
import pypsa
import os

print(os.getcwd())


def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and
    discount rate r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def prepare_costs(cost_file, params, nyears):
    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW", na=False), "value"] *= 1e3

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )

    costs = costs.fillna(params["fill_values"])

    def annuity_factor(v):
        return calculate_annuity(v["lifetime"], params["r"]) + v["FOM"] / 100

    costs["annuity_factor"] = costs.apply(annuity_factor, axis=1)

    costs["fixed"] = costs["annuity_factor"] * costs["investment"] * nyears

    return costs


if __name__ == "__main__":
    params = {
        "fill_values": 0.0,
        "r": 0.07,
        "nyears": 1,
        "year": 2025,
    }

    cost_file = f"cost_data/costs_{params['year']}.csv"

    costs = prepare_costs(cost_file, params, params["nyears"])

    print("Prepared cost table:")
    print(costs.head(), "\n")

    print("Columns:")
    print(costs.columns.tolist(), "\n")

    print("Technologies:")
    print(costs.index.tolist()[:20], "\n")

    # Example: inspect one technology
    if "onwind" in costs.index:
        print("Onshore wind data:")
        print(costs.loc["onwind"])