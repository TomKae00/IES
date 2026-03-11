import geopandas as gpd
import matplotlib.pyplot as plt

# -------------------------
# Onshore Netherlands
# -------------------------
countries = gpd.read_file("ne/ne_10m_admin_0_countries.shp").to_crs(4326)
nl_onshore = countries[countries["ADMIN"] == "Netherlands"].copy().dissolve()
onshore_shape = nl_onshore.geometry.iloc[0]

# -------------------------
# Offshore Netherlands EEZ
# -------------------------
eez = gpd.read_file("eez/eez.shp").to_crs(4326)

print(eez[["geoname", "territory1", "sovereign1", "area_km2"]])

nl_offshore = eez.dissolve()
offshore_shape = nl_offshore.geometry.iloc[0]
nl_offshore_metric = nl_offshore.to_crs(3035)

print(nl_offshore_metric.area / 1e6)# -------------------------
# Plot sanity check
# -------------------------
ax = nl_onshore.plot(figsize=(7, 7), alpha=0.5, edgecolor="black")
nl_offshore.boundary.plot(ax=ax)
plt.xlim(2.5, 7.5)
plt.ylim(50.5, 55.5)
plt.show()