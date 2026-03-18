import atlite

import logging

logging.basicConfig(level=logging.INFO)

cutout = atlite.Cutout(
    path="western-europe-2011-01.nc",
    module="era5",
    x=slice(-13.6913, 1.7712),
    y=slice(49.9096, 60.8479),
    time="2011-01",
)

cutout.prepare()

