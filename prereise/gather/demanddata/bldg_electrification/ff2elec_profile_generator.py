# This script creates time series for electricity loads from converting fossil fuel heating to electric heat pumps

### User inputs ###
yr_temps = (
    2016  # Year for which temperatures are used to compute loads; options are 2008-2017
)
bldg_class = "res"  # Building class for loads; options are (1) reidential ["res"] or (2) commercial ["com"]
hp_model = "advperfhp"  # Heat pump model to use. Options are (1) mid-performance cold climate heat pump ["midperfhp"], (2) advanced performance cold climate heat pump ["advperfhp"],(3) future performance heat pump ["futurehp"]
###################

# Import libraries
import os

import numpy as np
import pandas as pd

# Basic info and input files
## Lists
state_list = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
# COP and capacity ratio models based on:
## (a) 50th percentile NEEP CCHP database [midperfhp], (b) 90th percentile NEEP CCHP database [advperfhp], (c) future HP targets, average of residential and commercial targets [futurehp]
hp_param = pd.read_csv("reference_files/hp_parameters.csv")
puma_data = pd.read_csv("reference_files/puma_data.csv")
puma_slopes = pd.read_csv("reference_files/puma_slopes_{}.csv".format(bldg_class))

### midperfhp
def func_htg_cop_midperfhp(temp_c):
    temp_k = [i + 273.15 for i in temp_c]
    cop = [0] * len(temp_c)
    cop_base = [0] * len(temp_c)
    cr_base = [0] * len(temp_c)
    eaux_base = [0] * len(temp_c)

    pars = hp_param[hp_param["model"] == "midperfhp"].T
    T1_K = pars.iloc[3, 0]
    COP1 = pars.iloc[4, 0]
    T2_K = pars.iloc[8, 0]
    COP2 = pars.iloc[9, 0]
    T3_K = pars.iloc[13, 0]
    COP3 = pars.iloc[14, 0]
    CR3 = pars.iloc[15, 0]
    a = pars.iloc[16, 0]
    b = pars.iloc[17, 0]
    c = pars.iloc[18, 0]

    for i in range(len(temp_k)):
        if temp_k[i] + b > 0:
            cr_base[i] = a * np.log(temp_k[i]) + c
        if temp_k[i] > T2_K:
            cop_base[i] = ((COP1 - COP2) / (T1_K - T2_K)) * temp_k[i] + (
                COP2 * T1_K - COP1 * T2_K
            ) / (T1_K - T2_K)
        if temp_k[i] > T3_K and temp_k[i] <= T2_K:
            cop_base[i] = ((COP2 - COP3) / (T2_K - T3_K)) * temp_k[i] + (
                COP3 * T2_K - COP2 * T3_K
            ) / (T2_K - T3_K)
        if temp_k[i] <= T3_K:
            cop_base[i] = (cr_base[i] / CR3) * COP3

    eaux = [0.75 - i if 0.75 - i >= 0 else 0 for i in cr_base]

    sumlist = [
        (cr_base[i] + eaux[i]) / (cr_base[i] / cop_base[i] + eaux[i])
        if cr_base[i] != 0
        else 1
        for i in range(len(cr_base))
    ]
    cop = [
        1 if cr_base[i] == 0 else (1 if sumlist[i] < 1 else sumlist[i])
        for i in range(len(cr_base))
    ]
    return cop


### advperfhp
def func_htg_cop_advperfhp(temp_c):
    temp_k = [i + 273.15 for i in temp_c]
    cop = [0] * len(temp_c)
    cop_base = [0] * len(temp_c)
    cr_base = [0] * len(temp_c)
    eaux_base = [0] * len(temp_c)

    pars = hp_param[hp_param["model"] == "advperfhp"].T
    T1_K = pars.iloc[3, 0]
    COP1 = pars.iloc[4, 0]
    T2_K = pars.iloc[8, 0]
    COP2 = pars.iloc[9, 0]
    T3_K = pars.iloc[13, 0]
    COP3 = pars.iloc[14, 0]
    CR3 = pars.iloc[15, 0]
    a = pars.iloc[16, 0]
    b = pars.iloc[17, 0]
    c = pars.iloc[18, 0]

    for i in range(len(temp_k)):
        if temp_k[i] + b > 0:
            cr_base[i] = a * np.log(temp_k[i]) + c

        if temp_k[i] > T2_K:
            cop_base[i] = ((COP1 - COP2) / (T1_K - T2_K)) * temp_k[i] + (
                COP2 * T1_K - COP1 * T2_K
            ) / (T1_K - T2_K)
        if temp_k[i] > T3_K and temp_k[i] <= T2_K:
            cop_base[i] = ((COP2 - COP3) / (T2_K - T3_K)) * temp_k[i] + (
                COP3 * T2_K - COP2 * T3_K
            ) / (T2_K - T3_K)
        if temp_k[i] <= T3_K:
            cop_base[i] = (cr_base[i] / CR3) * COP3

    eaux = [0.75 - i if 0.75 - i >= 0 else 0 for i in cr_base]

    sumlist = [
        (cr_base[i] + eaux[i]) / (cr_base[i] / cop_base[i] + eaux[i])
        if cr_base[i] != 0
        else 1
        for i in range(len(cr_base))
    ]
    cop = [
        1 if cr_base[i] == 0 else (1 if sumlist[i] < 1 else sumlist[i])
        for i in range(len(cr_base))
    ]
    return cop


### futurehp
def func_htg_cop_futurehp(temp_c):
    temp_k = [i + 273.15 for i in temp_c]
    cop = [0] * len(temp_c)
    cop_base = [0] * len(temp_c)
    cr_base = [0] * len(temp_c)
    eaux_base = [0] * len(temp_c)

    pars = hp_param[hp_param["model"] == "futurehp"].T
    T1_K = pars.iloc[3, 0]
    COP1 = pars.iloc[4, 0]
    T2_K = pars.iloc[8, 0]
    COP2 = pars.iloc[9, 0]
    T3_K = pars.iloc[13, 0]
    COP3 = pars.iloc[14, 0]
    CR3 = pars.iloc[15, 0]
    a = pars.iloc[16, 0]
    b = pars.iloc[17, 0]
    c = pars.iloc[18, 0]

    for i in range(len(temp_k)):
        if temp_k[i] + b > 0:
            cr_base[i] = a * np.log(temp_k[i]) + c

        if temp_k[i] > T2_K:
            cop_base[i] = ((COP1 - COP2) / (T1_K - T2_K)) * temp_k[i] + (
                COP2 * T1_K - COP1 * T2_K
            ) / (T1_K - T2_K)
        if temp_k[i] > T3_K and temp_k[i] <= T2_K:
            cop_base[i] = ((COP2 - COP3) / (T2_K - T3_K)) * temp_k[i] + (
                COP3 * T2_K - COP2 * T3_K
            ) / (T2_K - T3_K)
        if temp_k[i] <= T3_K:
            cop_base[i] = (cr_base[i] / CR3) * COP3

    eaux = [0.75 - i if 0.75 - i >= 0 else 0 for i in cr_base]

    sumlist = [
        (cr_base[i] + eaux[i]) / (cr_base[i] / cop_base[i] + eaux[i])
        if cr_base[i] != 0
        else 1
        for i in range(len(cr_base))
    ]
    cop = [
        1 if cr_base[i] == 0 else (1 if sumlist[i] < 1 else sumlist[i])
        for i in range(len(cr_base))
    ]

    adv_cop = func_htg_cop_advperfhp(temp_c)
    cop_final = [
        cop_base[i] if cop_base[i] >= adv_cop[i] else adv_cop[i]
        for i in range(len(cop_base))
    ]
    return cop_final


# Reference temperatures for computations
temp_ref_res = 18.3
temp_ref_com = 16.7

if bldg_class == "com":
    temp_ref_it = temp_ref_com
else:
    temp_ref_it = temp_ref_res

# Loop through states to create profile outputs
for s in range(len(state_list)):
    state_it = state_list[s]

    # Load and subset relevant data for the state
    puma_data_it = puma_data[puma_data["state"] == state_it].reset_index()
    puma_slopes_it = puma_slopes[puma_slopes["state"] == state_it].reset_index()
    # temps_pumas_it = pd.read_csv("temps_pumas/temps_pumas_{}_{}.csv".format(state_it,yr_temps))
    temps_pumas_it = pd.read_csv(
        "https://besciences.blob.core.windows.net/datasets/pumas/temps_pumas_{}_{}.csv".format(
            state_it, yr_temps
        )
    )
    temps_pumas_transpose_it = temps_pumas_it.T

    # Load HP function
    func_htg_cop = globals()["func_htg_cop_{}".format(hp_model)]

    # Compute electric HP loads from fossil fuel conversion
    elec_htg_ff2hp_puma_mw_it_ref_temp = temps_pumas_transpose_it.applymap(
        lambda x: temp_ref_it - x if temp_ref_it - x >= 0 else 0
    )
    elec_htg_ff2hp_puma_mw_it_func = temps_pumas_transpose_it.apply(
        lambda x: np.reciprocal(func_htg_cop(x)), 1
    )
    elec_htg_ff2hp_puma_mw_it_func = pd.DataFrame(
        elec_htg_ff2hp_puma_mw_it_func.to_list()
    )
    elec_htg_ff2hp_puma_mw_it_func.index = list(
        elec_htg_ff2hp_puma_mw_it_ref_temp.index
    )

    elec_htg_ff2hp_puma_mw_it = elec_htg_ff2hp_puma_mw_it_ref_temp.multiply(
        elec_htg_ff2hp_puma_mw_it_func
    )

    pumalist = [
        puma_slopes_it["htg_slope_{}_btu_m2_degC".format(bldg_class)][i]
        * puma_data_it["{}_area_2010_m2".format(bldg_class)][i]
        * puma_data_it["frac_ff_sh_{}_2010".format(bldg_class)][i]
        * (293.0711 / (10 ** 6) / 1000)
        for i in range(len(puma_data_it))
    ]

    elec_htg_ff2hp_puma_mw_it = elec_htg_ff2hp_puma_mw_it.mul(pumalist, axis=0)
    elec_htg_ff2hp_puma_mw_it = elec_htg_ff2hp_puma_mw_it.T

    elec_htg_ff2hp_puma_mw_it.columns = temps_pumas_it.columns

    # Export profile file as CSV
    if os.path.exists("Profiles/"):
        elec_htg_ff2hp_puma_mw_it.to_csv(
            "Profiles/elec_htg_ff2hp_{}_{}_{}_{}_mw.csv".format(
                bldg_class, state_it, yr_temps, hp_model
            ),
            index=False,
        )
    else:
        os.makedirs("Profiles/")
        elec_htg_ff2hp_puma_mw_it.to_csv(
            "Profiles/elec_htg_ff2hp_{}_{}_{}_{}_mw.csv".format(
                bldg_class, state_it, yr_temps, hp_model
            ),
            index=False,
        )
