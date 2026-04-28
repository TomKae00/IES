# Integrated Energy Grids Course Project

This repository contains the PyPSA-based model and analysis scripts for the Integrated Energy Grids course project. The code is organized around a modular but simple workflow: model construction, scenario configuration, model execution, and task-specific analysis are kept separate.

## Repository structure

```text
project/
├── data/
│   ├── costs_2025.csv
│   ├── time_series_60min_singleindex_filtered_2015-2020.csv
│   └── heat_demand_cop.csv
│
├── results/
│   ├── networks/
│   ├── figures/
│   └── tables/
│
├── model/
│   ├── __init__.py
│   ├── model.py
│   ├── helpers.py
│   └── scenarios.py
│
├── analysis/
│   ├── __init__.py
│   ├── analysis_common.py
│   ├── analyze_base.py
│   ├── analyze_weather_sensitivity.py
│   ├── analyze_storage.py
│   ├── analyze_interconnected.py
│   ├── analyze_gas_network.py
│   ├── analyze_co2_sensitivity.py
│   ├── analyze_sector_coupling.py
│   └── analyze_policy_experiment.py
│
├── run_model.py
└── README.md
```

## Main workflow

The workflow is split into four parts:

1. `model/scenarios.py` defines what should be modeled.
2. `model/helpers.py` loads and prepares input data.
3. `model/model.py` builds the PyPSA network based on the selected scenario.
4. `run_model.py` selects a scenario, solves the model, and saves the optimized network.

Detailed post-processing is done separately in the `analysis/` folder.

## File overview

### `model/scenarios.py`

This file contains all scenario definitions and shared file paths.

It includes:

```python
FINANCIAL_PARAMETERS
FILE_PATHS
SCENARIOS
```

The scenario dictionary controls which model extensions are active. For example:

```python
"with_battery_storage": True,
"with_interconnectors": False,
"with_gas_network": False,
"with_heat_sector": False,
"co2_price": 0.0,
"co2_limit": None,
```

Typical scenarios include:

* `base`: single-country electricity model
* `storage`: single-country model with battery storage
* `interconnected`: regional electricity model with fixed interconnectors
* `gas`: regional model with CH4/H2 gas pipelines
* `sector_coupling`: electricity and heat co-optimization
* `co2_constraint`: model with a global CO2 cap

### `model/helpers.py`

This file contains generic helper functions that are not specific PyPSA components.

It includes functions such as:

```python
silence_gurobi_logger()
calculate_annuity()
prepare_costs()
load_country_timeseries()
load_all_countries_timeseries()
load_heat_timeseries()
calculate_conventional_marginal_cost()
```

The electricity time series are loaded with:

```python
load_all_countries_timeseries(...)
```

The heat-sector data are loaded with:

```python
load_heat_timeseries(...)
```

The heat-sector input file should contain columns such as:

```text
DK_heat_demand_total
DK_COP_ASHP_floor
DE_heat_demand_total
DE_COP_ASHP_floor
SE_heat_demand_total
SE_COP_ASHP_floor
```

Norwegian heat data are not available in the current heat input file. Therefore, Danish heat demand and air-source heat pump COP profiles are used as a proxy for Norway.

### `model/model.py`

This file contains the PyPSA model construction.

It does not define scenarios and does not load input files directly. Instead, it receives prepared input data and a scenario dictionary from `run_model.py`.

The main structure is:

```python
add_carriers()
set_carrier_co2_emissions()
add_electricity()
add_battery_storage()
add_interconnectors()
add_gas()
add_heat()
add_global_co2_constraint()
add_emission_prices()
create_network()
optimize_and_save_network()
print_model_summary()
```

The central function is:

```python
create_network(cost_data, all_timeseries_data, scenario, heat_timeseries=None)
```

It builds the network in layers:

```text
1. Add carriers
2. Add electricity system
3. Optionally add battery storage
4. Optionally add electricity interconnectors
5. Optionally add gas network
6. Optionally add heat sector
7. Optionally add CO2 price
8. Optionally add CO2 cap
```

### `run_model.py`

This is the main execution script.

It selects one scenario, loads the required input data, creates the PyPSA network, solves it, and saves the optimized network to `results/networks/`.

The active scenario is selected with:

```python
ACTIVE_SCENARIO = "base"
```

To run another model, change this to for example:

```python
ACTIVE_SCENARIO = "gas"
```

or:

```python
ACTIVE_SCENARIO = "sector_coupling"
```

Then run from the project root:

```bash
python run_model.py
```

The solved network is saved as:

```text
results/networks/{scenario_name}.nc
```

For example:

```text
results/networks/base_DK_2019.nc
results/networks/gas_network_2019.nc
results/networks/sector_coupling_heat_2019.nc
```

## Model components

### Electricity system

The electricity system is added with `add_electricity()`.

For each modeled country, it adds:

* electricity bus
* electricity demand
* solar PV
* onshore wind
* offshore wind
* CCGT, if no explicit gas network is modeled
* coal
* nuclear

When the gas network is enabled, CCGT is not added as a normal generator. Instead, CCGT is represented as a CH4-to-electricity conversion link inside `add_gas()`.

### Battery storage

Battery storage is added with `add_battery_storage()`.

Each battery consists of:

* battery bus
* energy store
* charging link
* discharging link

By default, batteries can be added to selected countries using the scenario setting:

```python
"battery_countries": ["DK"]
```

If this setting is not provided, batteries can be added to all modeled countries, depending on the implementation.

### Electricity interconnectors

Electricity interconnectors are added with `add_interconnectors()`.

The current network includes fixed interconnectors between Denmark, Germany, Sweden, and Norway. PyPSA `Line` components are bidirectional; `bus0` and `bus1` only define the positive flow direction.

The modeled corridors are:

```text
DK - NO
DK - SE
DK - DE
DE - SE
NO - SE
DE - NO
```

### Gas network

The gas network is added with `add_gas()`.

It includes:

* CH4 and H2 buses for each modeled country
* CCGT as CH4-to-electricity conversion
* electrolysis as electricity-to-H2 conversion
* optional H2 turbine
* CH4 supply in Norway
* CH4 and H2 pipelines between modeled countries

The CH4 supply is modeled as a generator on the Norwegian CH4 bus. Therefore, fossil gas emissions are assigned to the `CH4 supply` carrier.

### Heat sector coupling

The heat sector is added with `add_heat()`.

For each modeled country, it adds:

* heat bus
* heat demand
* air-source heat pump
* optional heat storage

Heat demand is loaded from:

```text
{country_code}_heat_demand_total
```

The air-source heat pump COP is loaded from:

```text
{country_code}_COP_ASHP_floor
```

For Norway, Danish profiles are used as a proxy because Norwegian heat data are not included in the input file.

## CO2 modeling

The model distinguishes between two CO2 policy instruments.

### CO2 price

A CO2 price increases the marginal costs of emitting technologies.

This is a price-based policy instrument:

```text
Input: CO2 price [EUR/tCO2]
Output: resulting emissions and system configuration
```

It is implemented using an emission price function similar to the approach used in PyPSA-Eur.

### CO2 cap

A CO2 cap imposes a global emissions limit.

This is a quantity-based policy instrument:

```text
Input: CO2 limit [tCO2]
Output: shadow price of the CO2 constraint and system configuration
```

The cap is added as a PyPSA `GlobalConstraint`.

For tasks f) and h), the CO2 cap should be used directly. To isolate the effect of the cap, the exogenous CO2 price should usually be set to zero:

```python
"co2_price": 0.0,
"co2_limit": selected_limit,
```

For task h), the required system-wide CO2 price is obtained as the shadow price of the global CO2 constraint.

## Running the model

From the project root, run:

```bash
python run_model.py
```

To change the scenario, edit `ACTIVE_SCENARIO` in `run_model.py`:

```python
ACTIVE_SCENARIO = "base"
```

Available scenarios are defined in `model/scenarios.py`.

## Running analyses

Analysis scripts are stored in the `analysis/` folder.

Example commands:

```bash
python -m analysis.analyze_base
python -m analysis.analyze_gas_network
python -m analysis.analyze_co2_sensitivity
python -m analysis.analyze_sector_coupling
```

Each analysis script should load one or several solved networks from:

```text
results/networks/
```

and write figures and tables to:

```text
results/figures/
results/tables/
```

## Suggested assignment mapping

```text
Task a: base electricity model
    Scenario: base
    Analysis: analyze_base.py

Task b: interannual weather sensitivity
    Scenario: base or storage run for several weather years
    Analysis: analyze_weather_sensitivity.py

Task c: storage model
    Scenario: storage
    Analysis: analyze_storage.py

Task d: interconnected electricity model
    Scenario: interconnected
    Analysis: analyze_interconnected.py

Task e: PTDF validation
    Scenario: interconnected
    Analysis: can be included in analyze_interconnected.py or a separate PTDF script

Task f: CO2 cap sensitivity
    Scenario: several CO2-constrained storage scenarios
    Analysis: analyze_co2_sensitivity.py

Task g: gas network
    Scenario: gas
    Analysis: analyze_gas_network.py

Task h: required CO2 price for selected decarbonisation target
    Scenario: selected CO2 cap scenario
    Analysis: analyze_co2_sensitivity.py or a dedicated CO2 price script

Task i: sector coupling
    Scenario: sector_coupling
    Analysis: analyze_sector_coupling.py

Task j: regional policy experiment
    Scenario: custom experiment scenario
    Analysis: analyze_policy_experiment.py
```

## Notes

* Keep `model/model.py` focused on PyPSA model construction.
* Keep data loading and cost preparation in `model/helpers.py`.
* Keep scenario definitions in `model/scenarios.py`.
* Keep all plotting and result interpretation outside the model file.
* Use clear result names so that each network, figure, and table can be traced back to a scenario.
