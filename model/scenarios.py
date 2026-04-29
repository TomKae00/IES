# =========================================================
# CONFIG
# =========================================================

FINANCIAL_PARAMETERS = {
    "fill_values": 0.0,
    "r": 0.07,
    "nyears": 1,
    "cost_year": 2025,
}

FILE_PATHS = {
    # Adjust these if your folders are named "Data" / "cost_data"
    "cost_file": f"data/costs_{FINANCIAL_PARAMETERS['cost_year']}.csv",
    "timeseries_file": "data/time_series_60min_singleindex_filtered_2015-2020.csv",
    "heat_file": "data/when2heat_filtered.csv",
    "network_output_dir": "results/networks",
}

SCENARIOS = {
    # -----------------------------------------------------
    # Part 1a: Base single-country model
    # -----------------------------------------------------
    "base": {
        "name": "base_DK",
        "weather_year": "2016",
        "countries": ["DK"],

        "with_battery_storage": False,
        "with_interconnectors": False,

        "with_ch4_network": False,
        "with_h2_network": False,

        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_price": 80.0,
        "co2_limit": None,
    },

    # -----------------------------------------------------
    # Part 1c: Storage model
    # -----------------------------------------------------
    "storage": {
        "name": "storage_DK",
        "weather_year": "2016",
        "countries": ["DK"],

        "with_battery_storage": True,
        "with_interconnectors": False,

        "with_ch4_network": False,
        "with_h2_network": False,

        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_price": 80.0,
        "co2_limit": None,
    },

    # -----------------------------------------------------
    # Part 1d: Interconnected electricity model
    # -----------------------------------------------------
    "interconnected": {
        "name": "interconnected",
        "weather_year": "2016",
        "countries": ["DK", "DE", "SE", "NO"],

        "with_battery_storage": False,
        "with_interconnectors": True,

        "with_ch4_network": False,
        "with_h2_network": False,

        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_price": 80.0,
        "co2_limit": None,
    },

    # -----------------------------------------------------
    # Part 2g: Gas network model
    # -----------------------------------------------------
    "gas_ch4_only": {
        "name": "gas_ch4_only",
        "weather_year": "2016",
        "countries": ["DK", "DE", "SE", "NO"],

        "with_battery_storage": True,
        "with_interconnectors": True,

        "with_ch4_network": True,
        "with_h2_network": False,

        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_price": 80.0,
        "co2_limit": None,
    },

    "gas_H2_only": {
        "name": "gas_H2_only",
        "weather_year": "2016",
        "countries": ["DK", "DE", "SE", "NO"],

        "with_battery_storage": True,
        "with_interconnectors": True,

        "with_ch4_network": False,
        "with_h2_network": True,

        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_price": 200.0,
        "co2_limit": None,
    },

    "gas_H2_only_co2_80": {
        "name": "gas_H2_only_co2_80",
        "weather_year": "2019",
        "countries": ["DK", "DE", "SE", "NO"],

        "with_battery_storage": True,
        "with_interconnectors": True,

        "with_ch4_network": False,
        "with_h2_network": True,

        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_price": 80.0,
        "co2_limit": None,
    },

    # -----------------------------------------------------
    # Part 2i: Sector-coupled heat model
    # -----------------------------------------------------
    "sector_coupling": {
        "name": "sector_coupling_heat",
        "weather_year": "2015",
        "countries": ["DK", "DE", "SE", "NO"],

        "with_battery_storage": False,
        "with_interconnectors": True,

        "with_ch4_network": True,
        "with_h2_network": False,

        "with_heat_sector": True,
        "with_heat_storage": True,

        "co2_price": 0.0,
        "co2_limit": 603779863.8428456 * 0.3,
    },

    # -----------------------------------------------------
    # Task g: CO2 price sensitivity (gas network analysis)
    # Run each scenario via run_model.py, then plot with
    # analyze_gas_network.plot_co2_sensitivity().
    # Saved as: sensitivity_ch4_co2_{price}_2019.nc / sensitivity_h2_co2_{price}_2019.nc
    # -----------------------------------------------------
    **{
        f"sensitivity_gas_ch4_only_co2_{price}": {
            "name": f"sensitivity_gas_ch4_only_co2_{price}",
            "weather_year": "2019",
            "countries": ["DK", "DE", "SE", "NO"],
            "with_battery_storage": True,
            "with_interconnectors": True,
            "with_ch4_network": True,
            "with_h2_network": False,
            "with_heat_sector": False,
            "with_heat_storage": False,
            "co2_price": float(price),
            "co2_limit": None,
        }
        for price in [0, 50, 100, 150, 200, 300]
    },

    **{
        f"sensitivity_gas_H2_only_co2_{price}": {
            "name": f"sensitivity_gas_H2_only_co2_{price}",
            "weather_year": "2019",
            "countries": ["DK", "DE", "SE", "NO"],
            "with_battery_storage": True,
            "with_interconnectors": True,
            "with_ch4_network": False,
            "with_h2_network": True,
            "with_heat_sector": False,
            "with_heat_storage": False,
            "co2_price": float(price),
            "co2_limit": None,
        }
        for price in [0, 50, 100, 150, 200, 300]
    },

    # -----------------------------------------------------
    # Task h: CO2 constrained model
    # -----------------------------------------------------
    "co2_cap": {
        "name": "co2_cap",
        "weather_year": "2016",
        "countries": ["DK", "DE", "SE", "NO"],

        "with_battery_storage": True,
        "with_interconnectors": True,

        "with_ch4_network": True,
        "with_h2_network": False,

        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_cap": 0.3,
        "co2_price": 0.0,
        "co2_limit": None,
    },
}