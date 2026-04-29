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

        "co2_price": 80.0,
        "co2_limit": None,
    },

    # -----------------------------------------------------
    # Example for Part 2f / 2h: CO2 constrained model
    # -----------------------------------------------------
    "co2_constraint": {
        "name": "co2_constraint",
        "weather_year": "2016",
        "countries": ["DK"],

        "with_battery_storage": False,
        "with_interconnectors": False,
        "with_gas_network": False,
        "with_h2_turbine": False,
        "with_heat_sector": False,
        "with_heat_storage": False,

        "co2_price": 80.0,

        # Unit should be tCO2 over the full modeled period.
        # Set this to the value you want to test.
        "co2_limit": 1_000_000.0,
    },
}