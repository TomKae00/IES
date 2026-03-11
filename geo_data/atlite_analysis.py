from pathlib import Path

import time
import atlite
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar


import logging

logging.basicConfig(level=logging.INFO)


# =========================================================
# 1) SETTINGS
# =========================================================
BASE_DIR = Path("..")
SHAPE_DIR = BASE_DIR / "geo_data"
CUTOUT_DIR = BASE_DIR / "geo_data" / "atlite_cutouts"
OUTPUT_DIR = BASE_DIR / "geo_data" / "capacity_factors"

COUNTRIES_FILE = SHAPE_DIR / "ne" / "ne_10m_admin_0_countries.shp"
EEZ_FILE = SHAPE_DIR / "eez" / "eez.shp"

CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# weather years for task b
YEARS = [2013]

DX = 0.5
DY = 0.5

# fixed Netherlands + North Sea box
BOUNDS = {
    "x": slice(2, 7.5),
    "y": slice(50.5, 56),
    "minx": 2.5,
    "miny": 50.5,
    "maxx": 7.5,
    "maxy": 55.5,
}

ONSHORE_TURBINE = "Vestas_V112_3MW"
OFFSHORE_TURBINE = "NREL_ReferenceTurbine_5MW_offshore"
SOLAR_PANEL = "CSi"
SOLAR_ORIENTATION = "latitude_optimal"


# =========================================================
# LOAD SHAPES
# =========================================================
def load_shapes():
    countries = gpd.read_file(COUNTRIES_FILE).to_crs(4326)
    eez = gpd.read_file(EEZ_FILE).to_crs(4326)

    # Netherlands onshore polygon
    nl_onshore = countries[countries["ADMIN"] == "Netherlands"].copy()
    nl_onshore = nl_onshore.dissolve()

    # Keep only the European Dutch EEZ part
    nl_offshore = eez.explode(index_parts=False).reset_index(drop=True)
    nl_offshore = nl_offshore[nl_offshore.centroid.y > 40].copy()
    nl_offshore = nl_offshore.dissolve()

    onshore_shape = nl_onshore.geometry.iloc[0]
    offshore_shape = nl_offshore.geometry.iloc[0]

    return nl_onshore, nl_offshore, onshore_shape, offshore_shape


# =========================================================
# PLOT SHAPES
# =========================================================
def plot_shapes(nl_onshore, nl_offshore):
    ax = nl_onshore.plot(figsize=(7, 7), alpha=0.5, edgecolor="black")
    nl_offshore.boundary.plot(ax=ax)

    plt.xlim(BOUNDS["minx"], BOUNDS["maxx"])
    plt.ylim(BOUNDS["miny"], BOUNDS["maxy"])
    plt.title("Netherlands onshore polygon and Dutch EEZ")
    plt.tight_layout()
    plt.show()


# =========================================================
# BUILD CUTOUT
# =========================================================
def build_cutout(year):
    cutout_path = CUTOUT_DIR / f"nl_era5_{year}_dx{DX}_dy{DY}.nc"

    cutout = atlite.Cutout(
        path=str(cutout_path),
        module="era5",
        x=BOUNDS["x"],
        y=BOUNDS["y"],
        time=str(year),
        dx=DX,
        dy=DY,
    )
    return cutout


# =========================================================
# PREPARE CUTOUT
# =========================================================
def prepare_cutout(cutout):
    start = time.time()
    with ProgressBar():
        cutout.prepare()
    end = time.time()
    print(f"Cutout preparation finished in {(end - start) / 60:.2f} minutes")


# =========================================================
# CALCULATE CAPACITY FACTORS
# =========================================================
def calculate_capacity_factors(cutout, onshore_shape, offshore_shape):
    cf_onshore = cutout.wind(
        turbine=ONSHORE_TURBINE,
        shapes=[onshore_shape],
        shapes_crs=4326,
        per_unit=True,
    )

    cf_offshore = cutout.wind(
        turbine=OFFSHORE_TURBINE,
        shapes=[offshore_shape],
        shapes_crs=4326,
        per_unit=True,
    )

    cf_solar = cutout.pv(
        panel=SOLAR_PANEL,
        orientation=SOLAR_ORIENTATION,
        shapes=[onshore_shape],
        shapes_crs=4326,
        per_unit=True,
    )

    cf_onshore = cf_onshore.squeeze().to_pandas()
    cf_offshore = cf_offshore.squeeze().to_pandas()
    cf_solar = cf_solar.squeeze().to_pandas()

    cf_onshore.name = "onshore_wind_cf"
    cf_offshore.name = "offshore_wind_cf"
    cf_solar.name = "solar_cf"

    cf = pd.concat([cf_onshore, cf_offshore, cf_solar], axis=1)
    cf = cf.clip(lower=0.0, upper=1.0)

    return cf


# =========================================================
# SAVE RESULTS
# =========================================================
def save_capacity_factors(cf, year):
    out_file = OUTPUT_DIR / f"capacity_factors_nl_{year}.csv"
    cf.to_csv(out_file)
    print(f"Saved: {out_file}")


# =========================================================
# RUN ONE YEAR
# =========================================================
def run_one_year(year, onshore_shape, offshore_shape):
    print(f"\n--- Running weather year {year} ---")
    print(
        f"Using bounds x={BOUNDS['x'].start:.2f}:{BOUNDS['x'].stop:.2f}, "
        f"y={BOUNDS['y'].start:.2f}:{BOUNDS['y'].stop:.2f}, "
        f"dx={DX}, dy={DY}"
    )

    cutout = build_cutout(year)
    prepare_cutout(cutout)

    print("Calculating capacity factors...")
    cf = calculate_capacity_factors(cutout, onshore_shape, offshore_shape)

    print("Saving results...")
    save_capacity_factors(cf, year)

    print("Mean capacity factors:")
    print(cf.mean())

    return cf


# =========================================================
# MAIN
# =========================================================
def main():
    nl_onshore, nl_offshore, onshore_shape, offshore_shape = load_shapes()

    print("Fixed cutout bounds:")
    print(BOUNDS)

    plot_shapes(nl_onshore, nl_offshore)

    for year in YEARS:
        run_one_year(year, onshore_shape, offshore_shape)


if __name__ == "__main__":
    main()