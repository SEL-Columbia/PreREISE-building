import time
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass
class Psm3Data:
    """Wrapper class for PSM3 data retrieved from NREL's API. Contains metadata
    from the first csv row and a data frame representing the remaining time series
    """

    lat: float
    lon: float
    tz: float
    elevation: float
    data_resource: pd.DataFrame

    def to_dict(self):
        """Convert the data to the format expected by nrel-pysam for running
        SAM simulations
        :return: (*dict*) -- a dictionary which can be passed to the pvwattsv7
        module
        """
        return {
            "lat": self.lat,
            "lon": self.lon,
            "tz": self.tz,
            "elev": self.elevation,
            "year": self.data_resource.index.year.tolist(),
            "month": self.data_resource.index.month.tolist(),
            "day": self.data_resource.index.day.tolist(),
            "hour": self.data_resource.index.hour.tolist(),
            "minute": self.data_resource.index.minute.tolist(),
            "dn": self.data_resource["DNI"].tolist(),
            "df": self.data_resource["DHI"].tolist(),
            "wspd": self.data_resource["Wind Speed"].tolist(),
            "tdry": self.data_resource["Temperature"].tolist(),
        }


class NrelApi:
    """Provides an interface to the NREL API for PSM3 data. It supports
    downloading this data in csv format, which we use to calculate solar output
    of a set of plants. The user will need to provide an API key.
    """

    def __init__(self, email, api_key, rate_limit=None):
        """Constructor"""
        if email is None:
            raise ValueError("Email is required")
        if api_key is None:
            raise ValueError("API key is required")

        self.email = email
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_at = None

    def _rate_limited(self, request_func):
        if self.rate_limit is None:
            return request_func()
        if self.last_request_at is None:
            self.last_request_at = time.time()
        remaining = time.time() - self.last_request_at
        if remaining < self.rate_limit:
            time.sleep(remaining)
        result = request_func()
        self.last_request_at = time.time()
        return result

    def _build_url(self, lat, lon, year="2016", leap_day=False):
        """Construct url with formatted query string for downloading psm3
        (physical solar model) data
        :param str lat: latitude of the plant
        :param str lon: longitude of the plant
        :param str year: the year
        :param bool leap_day: whether to use a leap day
        :return: (*str*) -- the url to download csv data
        """
        base_url = "https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv"
        payload = {
            "api_key": self.api_key,
            "names": year,
            "leap_day": str(leap_day).lower(),
            "interval": "60",
            "utc": "true",
            "email": self.email,
            "attributes": "dhi,dni,wind_speed,air_temperature",
            "wkt": f"POINT({lon}%20{lat})",
        }
        query = "&".join([f"{key}={value}" for key, value in payload.items()])
        return f"{base_url}?{query}"

    def get_psm3_at(self, lat, lon, dates=None, year="2016", leap_day=False):
        """Get PSM3 data at a given point for the specified year.
        :param str lat: latitude of the plant
        :param str lon: longitude of the plant
        :param str year: the year
        :param bool leap_day: whether to use a leap day
        :return: (*Psm3Data*) -- a data class containing metadata and time
        series for the given year and location
        """
        url = self._build_url(lat, lon, year, leap_day)
        info = self._rate_limited(lambda: pd.read_csv(url, nrows=1))
        tz, elevation = info["Local Time Zone"], info["Elevation"]

        data_resource = self._rate_limited(
            lambda: pd.read_csv(url, dtype=float, skiprows=2)
        )
        if dates is not None:
            data_resource.set_index(
                dates + timedelta(hours=int(tz.values[0])), inplace=True
            )
        return Psm3Data(lat, lon, float(tz), float(elevation), data_resource)
