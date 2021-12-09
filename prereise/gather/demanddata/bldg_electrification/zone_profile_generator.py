import numpy as np
import pandas as pd
import geopandas as gpd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def bkpt_scale(df, num_points, bkpt, heat_cool):
    """Adjust heating or cooling breakpoint to ensure there are enough data points to fit.

    :param pandas.DataFrame df: load and temperature for a certain hour of the day, wk or wknd.
    :param int num_points: minimum number of points required in df to fit.
    :param float bkpt: starting temperature breakpoint value.
    :param str heat_cool: dictates if breakpoint is shifted warmer for heating or colder for cooling

    :return: (*pandas.DataFrame*) dft -- adjusted dataframe filtered by new breakpoint. Original input df if size of initial df >= num_points
    :return: (*float*) bkpt -- updated breakpoint. Original breakpoint if size of initial df >= num_points
    """

    dft = df[df["temp_c"] <= bkpt].reset_index() if heat_cool == "heat" else df[df["temp_c"] >= bkpt].reset_index()
    if len(dft) < num_points:
        dft = df.sort_values(by=["temp_c"]).head(num_points).reset_index() if heat_cool == "heat" else df.sort_values(by=["temp_c"]).tail(num_points).reset_index()
        bkpt = dft["temp_c"][num_points - 1] if heat_cool == "heat" else dft["temp_c"][0]

    return dft.sort_index(), bkpt


def zone_shp_overlay(zone_name_shp):
    """Select pumas within a zonal load area

    :param str zone_name_shp: name of zone in BA_map.shp

    :return: (*pandas.DataFrame*) puma_data_zone -- puma data of all pumas within zone, including fraction within zone
    """

    shapefile = gpd.GeoDataFrame(gpd.read_file("shapefiles/BA_map.shp"))
    zone_shp = shapefile[shapefile["BA"] == zone_name_shp]
    pumas_shp = gpd.GeoDataFrame(gpd.read_file("shapefiles/pumas_overlay.shp"))

    puma_zone = gpd.overlay(pumas_shp, zone_shp.to_crs("EPSG:4269"))
    puma_zone["area"] = puma_zone["geometry"].to_crs({"proj":"cea"}).area 
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


def zonal_data(puma_data, hours_utc):
    """Aggregate puma metrics to population weighted hourly zonal values

    :param pandas.DataFrame puma_data: puma data within zone, output of zone_shp_overlay()
    :param pandas.DatetimeIndex hours_utc: index of UTC hours.

    :return: (*pandas.DataFrame*) temp_df -- hourly zonal values of temperature, wetbulb temperature, and darkness fraction
    """
    puma_pop_weights = (
        puma_data["pop_2010"] * puma_data["frac_in_zone"]
    ) / sum(puma_data["pop_2010"] * puma_data["frac_in_zone"])
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

    temp_df = pd.DataFrame(
        {
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

    return temp_df


def hourly_load_fit(load_temp_df):
    """Fit hourly heating, cooling, and baseload functions to load data

    :param pandas.DataFrame load_temp_df: hourly load and temperature data

    :return: (*pandas.DataFrame*) hourly_fits_df -- hourly and week/weekend breakpoints and coefficients for electricity use equations
    :return: (*float*) s_wb_db, i_wb_db -- slope and intercept of fit between dry and wet bulb temperatures of zone
    """
    

    def make_hourly_series(load_temp_df, i):
        daily_points = 8
        result = {}
        for wk_wknd in ["wk", "wknd"]:
            if wk_wknd == "wk":
                load_temp_hr = load_temp_df[
                    (load_temp_df["hour_local"] == i)
                    & (load_temp_df["weekday"] < 5)
                    & (load_temp_df["holiday"] == False)
                ].reset_index()
                numpoints = daily_points * 5
            elif wk_wknd == "wknd":
                load_temp_hr = load_temp_df[
                    (load_temp_df["hour_local"] == i)
                    & ((load_temp_df["weekday"] >= 5) | (load_temp_df["holiday"] == True))
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

            if s_dark < 0 or (max(load_temp_hr_heat['hourly_dark_frac']) - min(load_temp_hr_heat['hourly_dark_frac'])) < 0.3:
                s_dark, s_heat, i_heat = 0, s_heat_only, i_heat_only

            load_temp_hr_cool["cool_load_mw"] = [
                load_temp_hr_cool["load_mw"][j]
                - (s_heat * t_bph + i_heat)
                - s_dark * load_temp_hr_cool["hourly_dark_frac"][j]
                for j in range(len(load_temp_hr_cool))
            ]
            
            load_temp_hr_cool["temp_c_wb_diff"] = load_temp_hr_cool["temp_c_wb"] - (db_wb_fit[0]*load_temp_hr_cool["temp_c"]**2 + db_wb_fit[1]*load_temp_hr_cool["temp_c"] + db_wb_fit[2])

            lm_cool = LinearRegression().fit(
                np.array(
                    [
                        [
                            load_temp_hr_cool["temp_c"][i],
                            load_temp_hr_cool["temp_c_wb_diff"][i],
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
            
            t_bph = (
                -i_cool / s_cool_db
                if -i_cool / s_cool_db > t_bph
                else t_bph
            )
            result[wk_wknd] = {
                f"t.bpc.{wk_wknd}": t_bpc,
                f"t.bph.{wk_wknd}": t_bph,
                f"i.heat.{wk_wknd}": i_heat,
                f"s.heat.{wk_wknd}": s_heat,
                f"s.dark.{wk_wknd}": s_dark,
                f"i.cool.{wk_wknd}": i_cool,
                f"s.cool.{wk_wknd}.db": s_cool_db,
                f"s.cool.{wk_wknd}.wb": s_cool_wb,
            }
        return pd.Series({**result["wk"], **result["wknd"]})

    
    t_bpc_start = 10
    t_bph_start = 18.3

    db_wb_regr_df = load_temp_df[load_temp_df["temp_c"] >= t_bpc_start]
    
    db_wb_fit = np.polyfit(db_wb_regr_df["temp_c"], db_wb_regr_df["temp_c_wb"], 2)
    
    hourly_fits_df = pd.DataFrame([make_hourly_series(load_temp_df, i) for i in range(24)])

    return hourly_fits_df, db_wb_fit



def temp_to_energy(temp_series, hourly_fits_df, db_wb_fit):
    """Compute baseload, heating, and cooling electricity for a certain hour of year

    :param pandas.Series load_temp_series: data for the given hour.
    :param pandas.DataFrame hourly_fits_df: hourly and week/weekend breakpoints and
        coefficients for electricity use equations.
    :param float s_wb_db: slope of fit between dry and wet bulb temperatures of zone.
    :param float i_wb_db: intercept of fit between dry and wet bulb temperatures of zone.

    :return: (*list*) -- [baseload, heating, cooling]
    """
    temp = temp_series["temp_c"]
    temp_wb = temp_series["temp_c_wb"]
    dark_frac = temp_series["hourly_dark_frac"]
    zone_hour = temp_series["hour_local"]

    heat_eng = 0
    mid_cool_eng = 0
    cool_eng = 0

    wk_wknd = (
        "wk"
        if temp_series["weekday"] < 5 and temp_series["holiday"] == False
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
    ) = (
        hourly_fits_df.at[zone_hour, f"t.bpc.{wk_wknd}"],
        hourly_fits_df.at[zone_hour, f"t.bph.{wk_wknd}"],
        hourly_fits_df.at[zone_hour, f"i.heat.{wk_wknd}"],
        hourly_fits_df.at[zone_hour, f"s.heat.{wk_wknd}"],
        hourly_fits_df.at[zone_hour, f"s.dark.{wk_wknd}"],
        hourly_fits_df.at[zone_hour, f"i.cool.{wk_wknd}"],
        hourly_fits_df.at[zone_hour, f"s.cool.{wk_wknd}.db"],
        hourly_fits_df.at[zone_hour, f"s.cool.{wk_wknd}.wb"],

    )

    base_eng = s_heat * t_bph + s_dark * dark_frac + i_heat

    if temp <= t_bph:
        heat_eng = -s_heat * (t_bph - temp)

    if temp >= t_bph:
        cool_eng = s_cool_db * temp + s_cool_wb * (temp_wb - (db_wb_fit[0]*temp**2 + db_wb_fit[1]*temp + db_wb_fit[2])) + i_cool

    if temp > t_bpc and temp < t_bph:
                
        mid_cool_eng = ((temp - t_bpc) / (t_bph - t_bpc))**2 * (s_cool_db * t_bph + s_cool_wb * (temp_wb - (db_wb_fit[0]*temp**2 + db_wb_fit[1]*temp + db_wb_fit[2])) + i_cool)

    return [base_eng, heat_eng, max(cool_eng,0) + max(mid_cool_eng,0)]


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


def main(zone_name, zone_name_shp, load_year, year):
    """Run profile generator for one zone for one year.

    :param str zone_name: name of load zone used to save profile.
    :param str zone_name_shp: name of load zone within shapefile.
    :param int year: profile year to calculate.
    """
    zone_load = pd.read_csv(
        f"https://besciences.blob.core.windows.net/datasets/bldg_el/zone_loads_{year}/{zone_name}_demand_{year}_UTC.csv"
    )["demand.mw"]
    hours_utc_load_year = pd.date_range(
        start=f"{load_year}-01-01", end=f"{load_year+1}-01-01", freq="H", tz="UTC"
    )[:-1]
    
    hours_utc = pd.date_range(
        start=f"{year}-01-01", end=f"{year+1}-01-01", freq="H", tz="UTC"
    )[:-1]

    puma_data_zone = zone_shp_overlay(zone_name_shp)

    temp_df_load_year = zonal_data(puma_data_zone, hours_utc_load_year)
        
    temp_df = zonal_data(puma_data_zone, hours_utc)
    
    temp_df_load_year["load_mw"] = zone_load

    hourly_fits_df, db_wb_fit = hourly_load_fit(temp_df_load_year)
    hourly_fits_df.to_csv(f"dayhour_fits/{zone_name}_dayhour_fits_{year}.csv")


    zone_profile_load_MWh = pd.DataFrame({"hour_utc": list(range(len(hours_utc)))})
    energy_list = zone_profile_load_MWh.hour_utc.apply(
        lambda x: temp_to_energy(temp_df.loc[x], hourly_fits_df, db_wb_fit)
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
    zone_profile_load_MWh.to_csv(f"Profiles/{zone_name}_profile_load_mw_{year}.csv")

    plot_profile(zone_profile_load_MWh["total_load_mw"], zone_load)


if __name__ == "__main__":
    # Constants to be used when running this file as a script
    year = 2019
    load_year = 2019
    zone_name = "NYIS-ZOND"
    zone_name_shp = "North"
    main(zone_name, zone_name_shp, load_year, year)
