import os

import numpy as np
import pandas as pd

# This script creates time series for electricity loads from converting fossil fuel heating to electric heat pumps

# User inputs
# Year for which temperatures are used to compute loads; options are 2008-2017
yr_temps = 2016

# Building class for loads; options are (1) reidential ["res"] or (2) commercial ["com"]
bldg_class = "res"

# Heat pump model to use. Options are:
# (1) mid-performance cold climate heat pump ["midperfhp"],
# (2) advanced performance cold climate heat pump ["advperfhp"],
# (3) future performance heat pump ["futurehp"]
hp_model = "advperfhp"

# Basic info and input files
# Lists
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
# (a) 50th percentile NEEP CCHP database [midperfhp],
# (b) 90th percentile NEEP CCHP database [advperfhp],
# (c) future HP targets, average of residential and commercial targets [futurehp]
dir_path = os.path.dirname(os.path.abspath(__file__))
hp_param = pd.read_csv(os.path.join(dir_path, "data", "hp_parameters.csv"))
puma_data = pd.read_csv(os.path.join(dir_path, "data", "puma_data.csv"))
puma_slopes = pd.read_csv(
    os.path.join(dir_path, "data", f"puma_slopes_{bldg_class}.csv")
)


def calculate_cop(temp_c, model):
    cop_base, cr_base = _calculate_cop_base_cr_base(temp_c, model)

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


def _calculate_cop_base_cr_base(temp_c, model):
    temp_k = [i + 273.15 for i in temp_c]
    cop_base = [0] * len(temp_c)
    cr_base = [0] * len(temp_c)

    model_params = hp_param.set_index("model").loc[model]
    T1_K = model_params.loc["T1_K"]  # noqa: N806
    COP1 = model_params.loc["COP1"]  # noqa: N806
    T2_K = model_params.loc["T2_K"]  # noqa: N806
    COP2 = model_params.loc["COP2"]  # noqa: N806
    T3_K = model_params.loc["T3_K"]  # noqa: N806
    COP3 = model_params.loc["COP3"]  # noqa: N806
    CR3 = model_params.loc["CR3"]  # noqa: N806
    a = model_params.loc["a"]
    b = model_params.loc["b"]
    c = model_params.loc["c"]

    for i, temp in enumerate(temp_k):
        if temp + b > 0:
            cr_base[i] = a * np.log(temp) + c
        if temp > T2_K:
            cop_base[i] = ((COP1 - COP2) / (T1_K - T2_K)) * temp + (
                COP2 * T1_K - COP1 * T2_K
            ) / (T1_K - T2_K)
        if T3_K < temp <= T2_K:
            cop_base[i] = ((COP2 - COP3) / (T2_K - T3_K)) * temp + (
                COP3 * T2_K - COP2 * T3_K
            ) / (T2_K - T3_K)
        if temp <= T3_K:
            cop_base[i] = (cr_base[i] / CR3) * COP3

    return cop_base, cr_base


# midperfhp
def func_htg_cop_midperfhp(temp_c):
    return calculate_cop(temp_c, "midperfhp")


# advperfhp
def func_htg_cop_advperfhp(temp_c):
    return calculate_cop(temp_c, "advperfhp")


# futurehp
def func_htg_cop_futurehp(temp_c):
    cop_base, cr_base = _calculate_cop_base_cr_base(temp_c, "futurehp")

    adv_cop = func_htg_cop_advperfhp(temp_c)
    cop_final = [max(cop_base[i], adv_cop[i]) for i in range(len(cop_base))]
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

    temps_pumas_it = pd.read_csv(
        f"https://besciences.blob.core.windows.net/datasets/pumas/temps_pumas_{state_it}_{yr_temps}.csv"
    )
    temps_pumas_transpose_it = temps_pumas_it.T

    # Load HP function
    func_htg_cop = globals()[f"func_htg_cop_{hp_model}"]

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
        puma_slopes_it[f"htg_slope_{bldg_class}_btu_m2_degC"][i]
        * puma_data_it[f"{bldg_class}_area_2010_m2"][i]
        * puma_data_it[f"frac_ff_sh_{bldg_class}_2010"][i]
        * (293.0711 / (10 ** 6) / 1000)
        for i in range(len(puma_data_it))
    ]

    elec_htg_ff2hp_puma_mw_it = elec_htg_ff2hp_puma_mw_it.mul(pumalist, axis=0)
    elec_htg_ff2hp_puma_mw_it = elec_htg_ff2hp_puma_mw_it.T

    elec_htg_ff2hp_puma_mw_it.columns = temps_pumas_it.columns

    # Export profile file as CSV
    os.makedirs("Profiles", exist_ok=True)
    elec_htg_ff2hp_puma_mw_it.to_csv(
        f"Profiles/elec_htg_ff2hp_{bldg_class}_{state_it}_{yr_temps}_{hp_model}_mw.csv",
        index=False,
    )
