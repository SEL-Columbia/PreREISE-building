import numpy as np
import pandas as pd
import geopandas as gpd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

year = 2019
hours_utc = pd.date_range(
    start=f"{year}-01-01", end=f"{year+1}-01-01", freq="H", tz="UTC"
)[:-1]

load_name = "NYIS-ZONJ"
zone_name = "N.Y.C."
zone_load = pd.read_csv(
    f"https://besciences.blob.core.windows.net/datasets/bldg_el/zone_loads_{year}/{load_name}_demand_{year}_UTC.csv"
)["demand.mw"]


def bkpt_scale(df, num_points, bkpt, heat_cool):
    """Adjust heating or cooling breakpoint to ensure there are enough data points to fit.

    :param pandas.DataFrame df: load and temperature for a certain hour of the day, wk or wknd.
    :param int num_points: minimum number of points required in df to fit.
    :param float bkpt: starting temperature breakpoint value.
    :param str heat_cool: dictates if breakpoint is shifted warmer for heating or colder for cooling

    :return: (*pandas.DataFrame*) dft -- adjusted dataframe filtered by new breakpoint. Original input df if size of initial df >= num_points
    :return: (*float*) bkpt -- updated breakpoint. Original breakpoint if size of initial df >= num_points
    """

    dft = df[df["temp_c"] < bkpt] if heat_cool == "heat" else df[df["temp_c"] > bkpt]
    while len(dft) < num_points:
        bkpt = (bkpt + 0.1) if heat_cool == "heat" else (bkpt - 0.1)
        dft = (
            df[df["temp_c"] < bkpt] if heat_cool == "heat" else df[df["temp_c"] > bkpt]
        )
    return dft.reset_index(), bkpt


def zone_shp_overlay(zone_name):
    """Select pumas within a zonal load area

    :param str zone_name: name of zone in BA_map.shp

    :return: (*pandas.DataFrame*) puma_data_zone -- puma data of all pumas within zone, including fraction within zone
    """

    shapefile = gpd.GeoDataFrame(gpd.read_file("shapefiles/BA_map.shp"))
    zone_shp = shapefile[shapefile["BA"] == zone_name]
    pumas_shp = gpd.GeoDataFrame(gpd.read_file("shapefiles/pumas_overlay.shp"))

    puma_zone = gpd.overlay(pumas_shp, zone_shp.to_crs("EPSG:4269"))
    puma_zone["area"] = puma_zone.area
    puma_zone["puma"] = "puma_" + puma_zone["GEOID10"]

    puma_zone["area_frac"] = [
        puma_zone["area"][i]
        / list(pumas_shp[pumas_shp["puma"] == puma_zone["puma"][i]]["area"])[0]
        for i in range(len(puma_zone))
    ]

    puma_data_zone = pd.DataFrame(
        {"puma": puma_zone["puma"], "frac_in_zone": puma_zone["area_frac"]}
    )

    puma_data = pd.read_csv("data/puma_data.csv", index_col="puma")
    puma_data_zone = puma_data_zone.join(puma_data, on="puma")
    puma_data_zone = puma_data_zone.set_index("puma")

    return puma_data_zone


puma_data_zone = zone_shp_overlay(zone_name)


def zonal_data(puma_data):
    """Aggregate puma metrics to population weighted hourly zonal values

    :param pandas.DataFrame puma_data: puma data within zone, output of zone_shp_overlay()

    :return: (*pandas.DataFrame*) load_temp_df -- hourly zonal values of temperature, wetbulb temperature, and darkness fraction
    """
    puma_pop_weights = (
        puma_data_zone["pop_2010"] * puma_data_zone["frac_in_zone"]
    ) / sum(puma_data_zone["pop_2010"] * puma_data_zone["frac_in_zone"])
    zone_states = list(set(puma_data["state"]))
    timezone = max(
        set(list(puma_data["timezone"])), key=list(puma_data["timezone"]).count
    )

    puma_hourly_temps = pd.concat(
        list(
            pd.Series(data=zone_states).apply(
                lambda x: pd.read_csv(
                    f"https://besciences.blob.core.windows.net/datasets/bldg_el/pumas/temps/temps_pumas_{x}_{year}.csv"
                ).T
            )
        )
    )
    puma_hourly_temps_wb = pd.concat(
        list(
            pd.Series(data=zone_states).apply(
                lambda x: pd.read_csv(
                    f"https://besciences.blob.core.windows.net/datasets/bldg_el/pumas/temps_wetbulb/temps_wetbulb_pumas_{x}_{year}.csv"
                ).T
            )
        )
    )
    puma_hourly_dark_frac = pd.concat(
        list(
            pd.Series(data=zone_states).apply(
                lambda x: pd.read_csv(
                    f"https://besciences.blob.core.windows.net/datasets/bldg_el/pumas/dark_frac/dark_frac_pumas_{x}_{year}.csv"
                ).T
            )
        )
    )

    hours_local = hours_utc.tz_convert(timezone)
    is_holiday = pd.Series(hours_local).dt.date.isin(
        list(
            pd.Series(
                calendar().holidays(start=hours_local.min(), end=hours_local.max())
            ).dt.date
        )
    )

    load_temp_df = pd.DataFrame(
        {
            "load_mw": zone_load,
            "temp_c": puma_hourly_temps[puma_hourly_temps.index.isin(puma_data.index)]
            .mul(puma_pop_weights, axis=0)
            .sum(axis=0),
            "temp_c_wb": puma_hourly_temps_wb[
                puma_hourly_temps_wb.index.isin(puma_data.index)
            ]
            .mul(puma_pop_weights, axis=0)
            .sum(axis=0),
            "date_local": hours_local,
            "hour_local": hours_local.hour,
            "weekday": hours_local.weekday,
            "holiday": is_holiday,
            "hourly_dark_frac": puma_hourly_dark_frac[
                puma_hourly_dark_frac.index.isin(puma_data.index)
            ]
            .mul(puma_pop_weights, axis=0)
            .sum(axis=0),
        }
    )

    return load_temp_df


load_temp_df = zonal_data(puma_data_zone)


def hourly_load_fit(load_df):
    """Fit hourly heating, cooling, and baseload functions to load data

    :param pandas.DataFrame load_df: hourly load and temperature data

    :return: (*pandas.DataFrame*) hourly_fits_df -- hourly and week/weekend breakpoints and coefficients for electricity use equations
    :return: (*float*) s_wb_db, i_wb_db -- slope and intercept of fit between dry and wet bulb temperatures of zone
    """

    hourly_fits_df = pd.DataFrame(
        index=[[i for i in range(24)]],
        columns=[
            "t.bpc.wk",
            "t.bpc.wknd",
            "t.bph.wk",
            "t.bph.wknd",
            "i.heat.wk",
            "s.heat.wk",
            "s.dark.wk",
            "i.cool.wk",
            "i.cool.wk.db.only",
            "s.cool.wk.db",
            "s.cool.wk.wb",
            "s.cool.wk.db.only",
            "i.heat.wknd",
            "s.heat.wknd",
            "s.dark.wknd",
            "i.cool.wknd",
            "i.cool.wknd.db.only",
            "s.cool.wknd.db",
            "s.cool.wknd.wb",
            "s.cool.wknd.db.only",
        ],
    )
    t_bpc_start = 10
    t_bph_start = 18.3

    db_wb_regr_df = load_df[
        (load_df["temp_c"] >= t_bpc_start) & (load_df["temp_c"] <= t_bph_start)
    ]
    s_wb_db, i_wb_db, r_wb_db, pval_wb_db, stderr_wb_db = linregress(
        db_wb_regr_df["temp_c"], db_wb_regr_df["temp_c_wb"]
    )
    for wk_wknd in ["wk", "wknd"]:
        for i in range(len(hourly_fits_df)):
            daily_points = 8
            if wk_wknd == "wk":
                load_temp_hr = load_df[
                    (load_df["hour_local"] == i)
                    & (load_df["weekday"] < 5)
                    & (load_df["holiday"] == False)
                ].reset_index()
                numpoints = daily_points * 5
            elif wk_wknd == "wknd":
                load_temp_hr = load_df[
                    (load_df["hour_local"] == i)
                    & ((load_df["weekday"] >= 5) | (load_df["holiday"] == True))
                ].reset_index()
                numpoints = daily_points * 2

            load_temp_hr_heat, t_bpc = bkpt_scale(
                load_temp_hr, numpoints, t_bpc_start, "heat"
            )
            load_temp_hr_cool, t_bph = bkpt_scale(
                load_temp_hr, numpoints, t_bph_start, "cool"
            )

            lm_heat = LinearRegression().fit(
                np.array(
                    [
                        [
                            load_temp_hr_heat["temp_c"][j],
                            load_temp_hr_heat["hourly_dark_frac"][j],
                        ]
                        for j in range(len(load_temp_hr_heat))
                    ]
                ),
                load_temp_hr_heat["load_mw"],
            )
            s_heat, s_dark, i_heat = (
                lm_heat.coef_[0],
                lm_heat.coef_[1],
                lm_heat.intercept_,
            )

            s_heat_only, i_heat_only, r_heat, pval_heat, stderr_heat = linregress(
                load_temp_hr_heat["temp_c"], load_temp_hr_heat["load_mw"]
            )

            if s_dark < 0 or s_dark > 400:
                s_dark, s_heat, i_heat = 0, s_heat_only, i_heat_only

            load_temp_hr_cool["cool_load_mw"] = [
                load_temp_hr_cool["load_mw"][j]
                - (s_heat * t_bph + i_heat)
                - s_dark * load_temp_hr_cool["hourly_dark_frac"][j]
                for j in range(len(load_temp_hr_cool))
            ]

            lm_cool = LinearRegression().fit(
                np.array(
                    [
                        [
                            load_temp_hr_cool["temp_c"][i],
                            load_temp_hr_cool["temp_c_wb"][i],
                        ]
                        for i in range(len(load_temp_hr_cool))
                    ]
                ),
                load_temp_hr_cool["cool_load_mw"],
            )
            s_cool_db, s_cool_wb, i_cool = (
                lm_cool.coef_[0],
                lm_cool.coef_[1],
                lm_cool.intercept_,
            )
            s_cool_db_only, i_cool_db_only, r_cool, pval_cool, stderr_cool = linregress(
                load_temp_hr_cool["temp_c"], load_temp_hr_cool["cool_load_mw"]
            )

            t_bph = (
                -i_cool_db_only / s_cool_db_only
                if -i_cool_db_only / s_cool_db_only > t_bph
                else t_bph
            )

            hourly_fits_df.loc[
                i,
                [
                    f"t.bpc.{wk_wknd}",
                    f"t.bph.{wk_wknd}",
                    f"i.heat.{wk_wknd}",
                    f"s.heat.{wk_wknd}",
                    f"s.dark.{wk_wknd}",
                    f"i.cool.{wk_wknd}",
                    f"i.cool.{wk_wknd}.db.only",
                    f"s.cool.{wk_wknd}.db",
                    f"s.cool.{wk_wknd}.wb",
                    f"s.cool.{wk_wknd}.db.only",
                ],
            ] = [
                t_bpc,
                t_bph,
                i_heat,
                s_heat,
                s_dark,
                i_cool,
                i_cool_db_only,
                s_cool_db,
                s_cool_wb,
                s_cool_db_only,
            ]

    return hourly_fits_df, s_wb_db, i_wb_db


hourly_fits_df, s_wb_db, i_wb_db = hourly_load_fit(load_temp_df)


def temp_to_energy(temp, temp_wb, dark_frac, hour):
    """Compute baseload, heating, and cooling electricity for a certain hour of year

    :param float temp: drybulb temperature
    :param float temp_wb: wetbulb temperature
    :param float dark_frac: darkness fraction
    :param int hour: hour of year

    :return: (*list*) -- [baseload, heating, cooling]
    """

    zone_hour = load_temp_df["hour_local"][hour]
    heat_eng = 0
    cool_cool_eng = 0
    cool_hot_humid_eng = 0
    cool_hot_dry_eng = 0
    wk_wknd = (
        "wk"
        if load_temp_df["weekday"][hour] < 5 and load_temp_df["holiday"][hour] == False
        else "wknd"
    )

    (
        t_bpc,
        t_bph,
        i_heat,
        s_heat,
        s_dark,
        i_cool,
        s_cool_db,
        s_cool_wb,
        i_cool_db_only,
        s_cool_db_only,
    ) = (
        hourly_fits_df.at[zone_hour, f"t.bpc.{wk_wknd}"][0],
        hourly_fits_df.at[zone_hour, f"t.bph.{wk_wknd}"][0],
        hourly_fits_df.at[zone_hour, f"i.heat.{wk_wknd}"][0],
        hourly_fits_df.at[zone_hour, f"s.heat.{wk_wknd}"][0],
        hourly_fits_df.at[zone_hour, f"s.dark.{wk_wknd}"][0],
        hourly_fits_df.at[zone_hour, f"i.cool.{wk_wknd}"][0],
        hourly_fits_df.at[zone_hour, f"s.cool.{wk_wknd}.db"][0],
        hourly_fits_df.at[zone_hour, f"s.cool.{wk_wknd}.wb"][0],
        hourly_fits_df.at[zone_hour, f"i.cool.{wk_wknd}.db.only"][0],
        hourly_fits_df.at[zone_hour, f"s.cool.{wk_wknd}.db.only"][0],
    )

    base_eng = s_heat * t_bph + s_dark * dark_frac + i_heat

    if temp <= t_bph:
        heat_eng = -s_heat * (t_bph - temp)

    if temp >= t_bph and temp_wb >= i_wb_db + s_wb_db * t_bph:
        cool_hot_humid_eng = s_cool_db * temp + s_cool_wb * temp_wb + i_cool

    if temp >= t_bph and temp_wb < i_wb_db + s_wb_db * t_bph:
        cool_hot_dry_eng = s_cool_db_only * temp + i_cool_db_only

    if temp > t_bpc and temp < t_bph:
        cool_cool_eng = (
            (
                (i_cool_db_only + s_cool_db_only * t_bph)
                / (s_cool_db_only * (t_bph - t_bpc))
            )
            * ((temp - t_bpc) ** 2 / (t_bph - t_bpc))
            * s_cool_db_only
        )

    return [base_eng, heat_eng, cool_hot_humid_eng + cool_hot_dry_eng + cool_cool_eng]


zone_profile_load_MWh = pd.DataFrame({"hour_utc": list(range(len(hours_utc)))})
energy_list = zone_profile_load_MWh.hour_utc.apply(
    lambda x: temp_to_energy(
        load_temp_df["temp_c"][x],
        load_temp_df["temp_c_wb"][x],
        load_temp_df["hourly_dark_frac"][x],
        x,
    )
)
(
    zone_profile_load_MWh["base_load_mw"],
    zone_profile_load_MWh["heat_load_mw"],
    zone_profile_load_MWh["cool_load_mw"],
    zone_profile_load_MWh["total_load_mw"],
) = (
    energy_list.apply(lambda x: x[0]),
    energy_list.apply(lambda x: x[1]),
    energy_list.apply(lambda x: x[2]),
    energy_list.apply(lambda x: sum(x)),
)
zone_profile_load_MWh = zone_profile_load_MWh.set_index("hour_utc")
zone_profile_load_MWh.to_csv(f"Profiles/{load_name}_profile_load_mw_{year}.csv")


def plot_profile(profile, actual):
    """Plot profile vs. actual load

    :param pandas.Series profile: total profile hourly load
    :param pandas.Series actual: zonal hourly load data

    :return: (*plot*)
    """

    mrae = np.mean(
        [np.abs(profile[i] - actual[i]) / actual[i] for i in range(len(profile))]
    )

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(list(profile.index), profile)
    plt.plot(list(actual.index), actual)
    plt.legend(["Profile", "Actual"])
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title(
        "Zone "
        + zone_name
        + " Load Comparison \n"
        + "mrae = "
        + str(round(mrae * 100, 2))
        + "%"
    )


plot_profile(zone_profile_load_MWh["total_load_mw"], zone_load)
