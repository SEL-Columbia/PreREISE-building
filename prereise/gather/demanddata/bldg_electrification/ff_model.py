import os

import pandas as pd
import numpy as np

from scipy.optimize import least_squares

from prereise.gather.demanddata.bldg_electrification import const

# 2010 ff data used for fitting
start_date = "2010-01-01"
end_date = "2011-01-01"
yr_temps = 2010

# Computing number of hours each month for normalization in the fitting code below
dt = pd.date_range(start = start_date, end = end_date, freq='H').to_pydatetime().tolist()
datetimes = pd.DataFrame({'date_time': [dt[i].strftime('%Y-%m-%d %H:%M:%S') for i in range(len(dt)-1)]})
month_hrly = [int(dt[i].strftime('%m')) for i in range(len(dt)-1)]

# Load in historical 2010 fossil fuel usage data
dir_path = os.path.dirname(os.path.abspath(__file__))
ng_usage_data_res = pd.read_csv(os.path.join(dir_path, "data", "ng_monthly_mmbtu_2010_res.csv"))
ng_usage_data_com = pd.read_csv(os.path.join(dir_path, "data", "ng_monthly_mmbtu_2010_com.csv"))
fok_usage_data = pd.read_csv(os.path.join(dir_path, "data", "fok_data_bystate_2010.csv"))
othergas_usage_data = pd.read_csv(os.path.join(dir_path, "data", "propane_data_bystate_2010.csv"))

puma_data = pd.read_csv(os.path.join(dir_path, "data", "puma_data.csv"), index_col=False)

# Initialize dataframes to store state heating slopes
state_slopes_res = pd.DataFrame(columns=(['state', 'r2', 'sh_slope', 'dhw_const', 'dhw_slope', 'other_const']))
state_slopes_com = pd.DataFrame(columns=(['state', 'r2', 'sh_slope', 'dhw_const', 'cook_const', 'other_const', 'other_slope']))

for state in const.state_list:
    # Load puma data
    puma_data_it = puma_data[puma_data['state'] == state].reset_index()
    n_tracts = len(puma_data_it)

    # Load puma temperatures
    temps_pumas = temps_pumas = pd.read_csv(f"https://besciences.blob.core.windows.net/datasets/pumas/temps_pumas_{state}_{yr_temps}.csv")
    temps_pumas_transpose = temps_pumas.T

    for clas in const.classes:
        
        temp_ref_it = const.temp_ref_res if clas == 'res' else const.temp_ref_com
        
        # percentage of puma area that uses fossil fuel        
        frac_ff_sh = puma_data_it[f'frac_ff_sh_{clas}_2010']
        frac_ff_dhw = puma_data_it[f'frac_ff_dhw_{clas}_2010']
        frac_ff_cook = puma_data_it['frac_ff_cook_com_2010']
        if clas == 'res':
            frac_ff_other = puma_data_it['frac_ff_other_res_2010']
        else:
            frac_ff_other = puma_data_it['frac_ff_sh_com_2010']
        
        # puma area * percentage
        areas_ff_sh_it = [puma_data_it[f'{clas}_area_2010_m2'] * frac_ff_sh]
        areas_ff_dhw_it = [puma_data_it[f'{clas}_area_2010_m2'] * frac_ff_dhw]
        areas_ff_other_it = [puma_data_it[f'{clas}_area_2010_m2'] * frac_ff_other]
        areas_ff_cook_it = [puma_data_it[f'{clas}_area_2010_m2'] * frac_ff_cook]
        
        # sum of previous areas to be used in fitting 
        sum_areaff_sh = sum(areas_ff_sh_it)
        sum_areaff_dhw = sum(areas_ff_dhw_it)
        sum_areaff_other = sum(areas_ff_other_it)
        sum_areaff_cook = sum(areas_ff_cook_it)

        # Load monthly natural gas usage for the state
        natgas = list(ng_usage_data_res[state] if clas == 'res' else ng_usage_data_com[state])
        sum_natgas = sum(natgas)
        # Load annual fuel oil/kerosene and other gas/propane usage for the state
        fok = list(fok_usage_data[fok_usage_data['state'] == state][f'fok.{clas}.mmbtu'])[0]
        other = list(othergas_usage_data[othergas_usage_data['state'] == state][f'propane.{clas}.mmbtu'])[0]
        totfuel = fok+sum_natgas+other
        # Scale total fossil fuel usage by monthly natural gas
        ff_usage_data_it = [natgas[i]/sum_natgas * (totfuel) for i in range(len(natgas))]
                
        temp_ref_lin_dec_it = temp_ref_it
        temp_ref_lin_inc_it = temp_ref_it
    
        if clas == "res":
                        
            # Hourly heating degrees for all pumas in a given state, multiplied by their corresponding area and percent fossil fuel, summed up to one hourly list
            hd_hourly_it_sh = temps_pumas_transpose.applymap(lambda x: temp_ref_lin_dec_it-x if temp_ref_lin_dec_it-x >= 0 else 0).mul(areas_ff_sh_it, axis=0).sum(axis=0)
            hd_hourly_it_dhw = temps_pumas_transpose.applymap(lambda x: temp_ref_lin_dec_it-x).mul(areas_ff_dhw_it, axis=0).sum(axis=0)
    
            # Average hd per month
            df_hourly_it = pd.DataFrame({'month':month_hrly, 'hd_hourly_it_sh':hd_hourly_it_sh, 'hd_hourly_it_dhw':hd_hourly_it_dhw})
            hd_group_sh = df_hourly_it['hd_hourly_it_sh'].groupby(df_hourly_it['month'])
            hd_group_dhw = df_hourly_it['hd_hourly_it_dhw'].groupby(df_hourly_it['month'])
            df_monthly_it = pd.DataFrame({'hd_avg_sh': hd_group_sh.mean(), 'hd_avg_dhw': hd_group_dhw.mean(), 'n_hrs': hd_group_sh.count()})
            # Fossil fuel average monthly mmbtu, normalized by hours in month
            df_monthly_it['ff_monthly_mmbtu'] = [ff_usage_data_it[i]/list(df_monthly_it['n_hrs'])[i] for i in range(len(df_monthly_it))]

            # Fitting function: Returns difference between fitted equation and actual fossil fuel usage for the least_squares function to minimize
            def func_r(par, sh, dhw, ff):
                err = ff - (par[0]*sh + (sum_areaff_dhw*par[1]+const.dhw_lin_scalar*par[1]*dhw) + sum_areaff_other*par[2])
                return err
            
            # Input data points for fitting
            data_sh = np.array(df_monthly_it['hd_avg_sh'])
            data_dhw = np.array(df_monthly_it['hd_avg_dhw'])
            data_ff = df_monthly_it['ff_monthly_mmbtu']

            # Least squares solver
            lm_it = least_squares(func_r, const.bounds_lower_res, args=(data_sh, data_dhw, data_ff), bounds=(const.bounds_lower_res,const.bounds_upper_res))
            
            # Solved coefficients for slopes and constants
            par_sh_l = lm_it.x[0] 
            par_dhw_c = lm_it.x[1] 
            par_dhw_l = lm_it.x[1] * const.dhw_lin_scalar
            par_other_c = lm_it.x[2] 
            
            # Calculate r2 value of fit
            residuals = list(lm_it.fun)
            sumres = 0
            sumtot = 0
            for i in range(len(list(data_ff))):
                sumres += residuals[i]**2
                sumtot += (list(data_ff)[i]-np.mean(data_ff))**2
            r2 = 1 - (sumres/sumtot)
            
            # Add coefficients to output dataframe
            df_i = len(state_slopes_res)
            state_slopes_res.loc[df_i] = [state, r2, par_sh_l, par_dhw_c, par_dhw_l, par_other_c]
                 
        else: 
            
            # Hourly heating degrees for all pumas in a given state, multiplied by their corresponding area and percent fossil fuel, summed up to one hourly list
            hd_hourly_it_sh = temps_pumas_transpose.applymap(lambda x: temp_ref_lin_dec_it-x if temp_ref_lin_dec_it-x >= 0 else 0).mul(areas_ff_sh_it, axis=0).sum(axis=0)
            hd_hourly_it_other = temps_pumas_transpose.applymap(lambda x: x-temp_ref_lin_inc_it if x-temp_ref_lin_inc_it >= 0 else 0).mul(areas_ff_other_it, axis=0).sum(axis=0)

            # Average hd per month
            df_hourly_it = pd.DataFrame({'month':month_hrly, 'hd_hourly_it_sh':hd_hourly_it_sh, 'hd_hourly_it_other':hd_hourly_it_other})
            hd_group_sh = df_hourly_it['hd_hourly_it_sh'].groupby(df_hourly_it['month'])
            hd_group_other = df_hourly_it['hd_hourly_it_other'].groupby(df_hourly_it['month'])
            df_monthly_it = pd.DataFrame({'hd_avg_sh': hd_group_sh.mean(), 'hd_avg_other': hd_group_other.mean(), 'n_hrs': hd_group_sh.count()})
            # Fossil fuel average monthly mmbtu, normalized by hours in month
            df_monthly_it['ff_monthly_mmbtu'] = [ff_usage_data_it[i]/list(df_monthly_it['n_hrs'])[i] for i in range(len(df_monthly_it))]
            
            # Fitting function: Returns difference between fitted equation and actual fossil fuel usage for the least_squares function to minimize
            def func_c(par, sh, other, ff):  
                err = ff - (par[0]*sh + par[1]*sum_areaff_dhw + const.cook_c_scalar*par[1]*sum_areaff_cook + (sum_areaff_other*par[2]+par[3]*other))
                return err
            
            # Input data points for fitting
            data_sh = np.array(df_monthly_it['hd_avg_sh'])
            data_other = np.array(df_monthly_it['hd_avg_other'])
            data_ff = df_monthly_it['ff_monthly_mmbtu']
            
            # Least squares solver
            lm_it = least_squares(func_c, const.bounds_lower_com, args=(data_sh, data_other, data_ff), bounds=(const.bounds_lower_com, const.bounds_upper_com))
            
            # Solved coefficients for slopes and constants
            par_sh_l = lm_it.x[0] 
            par_dhw_c = lm_it.x[1] 
            par_cook_c = lm_it.x[1] * const.cook_c_scalar 
            par_other_c = lm_it.x[2]
            par_other_l = lm_it.x[3]

            # Calculate r2 value of fit
            residuals = list(lm_it.fun)
            sumres = 0
            sumtot = 0
            for i in range(len(list(data_ff))):
                sumres += residuals[i]**2
                sumtot += (list(data_ff)[i]-np.mean(data_ff))**2
            r2 = 1 - (sumres/sumtot)
            
            # Add coefficients to output dataframe
            df_i = len(state_slopes_com)
            state_slopes_com.loc[df_i] = [state, r2, par_sh_l, par_dhw_c, par_cook_c, par_other_c, par_other_l]
 
# Export heating/hot water/cooking coefficients for each state            
state_slopes_res.to_csv(os.path.join(dir_path, "data", "state_slopes_ff_res.csv"), index=False)
state_slopes_com.to_csv(os.path.join(dir_path, "data", "state_slopes_ff_com.csv"), index=False)

##############################################
# Space heating slope adjustment for climate #
##############################################

# Create data frames for space heating fossil fuel usage slopes at each PUMA
puma_slopes_res = pd.DataFrame(columns=(['state', 'puma', 'htg_slope_res_mmbtu_m2_degC']))
puma_slopes_com = pd.DataFrame(columns=(['state', 'puma', 'htg_slope_com_mmbtu_m2_degC']))

for i in range(len(puma_data)):
    state_it = puma_data['state'][i]
    
    df_index_res = len(puma_slopes_res)
    puma_slopes_res.loc[df_index_res] = [state_it, puma_data['puma'][i], list(state_slopes_res[state_slopes_res['state'] == state_it]['sh_slope'])[0]]
    
    df_index_com = len(puma_slopes_com)
    puma_slopes_com.loc[df_index_com ] = [state_it, puma_data['puma'][i], list(state_slopes_com[state_slopes_com['state'] == state_it]['sh_slope'])[0]]

puma_data['hd_183C_2010'] = '' 
puma_data['hd_167C_2010'] = '' 

for state in const.state_list:

    # Load puma temperatures
    temps_pumas = pd.read_csv(f"https://besciences.blob.core.windows.net/datasets/pumas/temps_pumas_{state}_{yr_temps}.csv")
    temps_pumas_transpose = temps_pumas.T

    # Hourly temperature difference below const.temp_ref_res/com for each puma
    temp_diff_res = temps_pumas_transpose.applymap(lambda x: const.temp_ref_res-x if const.temp_ref_res-x >= 0 else 0).T
    temp_diff_com = temps_pumas_transpose.applymap(lambda x: const.temp_ref_com-x if const.temp_ref_com-x >= 0 else 0).T

    # Annual heating degrees for each puma
    for i in list(puma_data[puma_data['state'] == state]['puma']):
        puma_data.at[puma_data.loc[puma_data['puma'] == i].index[0], 'hd_183C_2010'] = sum(list(temp_diff_res[i]))
        puma_data.at[puma_data.loc[puma_data['puma'] == i].index[0], 'hd_167C_2010'] = sum(list(temp_diff_com[i]))

# Load in state groups consistent with building area scale adjustments
area_scale_res = pd.read_csv(os.path.join(dir_path, "data", "area_scale_res.csv"), index_col=False)
area_scale_com = pd.read_csv(os.path.join(dir_path, "data", "area_scale_com.csv"), index_col=False)

# Extract res state groups from area_scale_res
res_state_groups = []
for i in area_scale_res.T.columns:
    stategrp = []
    for j in range(5):
        if len(str(area_scale_res.T[i][j])) == 2:
            stategrp.append(area_scale_res.T[i][j])
    res_state_groups.append(stategrp)
 
# Extract com state groups from area_scale_com  
com_state_groups = []
for i in area_scale_com.T.columns:
    stategrp = []
    for j in range(9):
        if len(str(area_scale_com.T[i][j])) == 2:
            stategrp.append(area_scale_com.T[i][j])
    com_state_groups.append(stategrp)

hdd65listres = []
htgslppoplistres = []
for res_state_group in res_state_groups:
    # List of pumas in each state group
    pumas_it = list(puma_data[puma_data['state'].isin(res_state_group)]['puma']) 
        
    # Population weighted heating degree days
    hdd65listres.append(sum(list(map(lambda x: sum(puma_data[puma_data['puma'] == x]['hdd65_normals_2010'] * puma_data[puma_data['puma'] == x]['pop_2010'])/sum(puma_data[puma_data['puma'].isin(pumas_it)]['pop_2010']), pumas_it))))
    # Population and heating degree day weighted heating slopes
    htgslppoplistres.append(sum(list(map(lambda x: sum(puma_slopes_res[puma_slopes_res['puma'] == x]['htg_slope_res_mmbtu_m2_degC'] * puma_data[puma_data['puma'] == x]['hdd65_normals_2010'] * puma_data[puma_data['puma'] == x]['pop_2010'])/sum(puma_data[puma_data['puma'].isin(pumas_it)]['pop_2010'] * puma_data[puma_data['puma'].isin(pumas_it)]['hdd65_normals_2010']), pumas_it))))
 
area_scale_res['hdd_normals_2010_popwtd'] = hdd65listres
area_scale_res['htg_slope_res_mmbtu_m2_degC_pophddwtd'] = htgslppoplistres

hdd65listcom = []
htgslppoplistcom = []
for com_state_group in com_state_groups:
    # Commercial equivalents of the previous for loop
    pumas_it = list(puma_data[puma_data['state'].isin(com_state_group)]['puma']) 
    hdd65listcom.append(sum(list(map(lambda x: sum(puma_data[puma_data['puma'] == x]['hdd65_normals_2010'] * puma_data[puma_data['puma'] == x]['pop_2010'])/sum(puma_data[puma_data['puma'].isin(pumas_it)]['pop_2010']), pumas_it))))
    htgslppoplistcom.append(sum(list(map(lambda x: sum(puma_slopes_com[puma_slopes_com['puma'] == x]['htg_slope_com_mmbtu_m2_degC'] * puma_data[puma_data['puma'] == x]['hdd65_normals_2010'] * puma_data[puma_data['puma'] == x]['pop_2010'])/sum(puma_data[puma_data['puma'].isin(pumas_it)]['pop_2010'] * puma_data[puma_data['puma'].isin(pumas_it)]['hdd65_normals_2010']), pumas_it)))) #check sumproduct denom

area_scale_com['hdd_normals_2010_popwtd'] = hdd65listcom
area_scale_com['htg_slope_com_mmbtu_m2_degC_pophddwtd'] = htgslppoplistcom

# Interpolating 2010 areas from the two survey years provided
area_scale_res['2010_RECS'] = area_scale_res['RECS2009'] + (area_scale_res['RECS2015']-area_scale_res['RECS2009'])/6
area_scale_com['2010_CBECS'] = area_scale_com['CBECS2012'] - (area_scale_com['CBECS2012']-area_scale_com['CBECS2003'])*(2/9)

# Divide by 1000 for robust solver
area_scale_res['hdd_normals_2010_popwtd_div1000'] = area_scale_res['hdd_normals_2010_popwtd']/1000
area_scale_com['hdd_normals_2010_popwtd_div1000'] = area_scale_com['hdd_normals_2010_popwtd']/1000

# Minimize error between actual slopes and fitted function
## Note for fitting to converge, hdd must be divided by 1000 and slopes in btu
def model(par, hdd_div1000, slope_btu):
    err = slope_btu - (par[0] + par[1] * (1-np.exp(-par[2] * hdd_div1000)))/hdd_div1000
    return err

# Least_squares residential model, to solve slope = (a + b*(1 - exp(-c*hdd)))/hdd    
ls_res = least_squares(model, [35,35,0.8], args=(np.array(area_scale_res['hdd_normals_2010_popwtd_div1000']), np.array(area_scale_res['htg_slope_res_mmbtu_m2_degC_pophddwtd'])*10**6), method='trf', loss='soft_l1', f_scale=0.1, bounds=(0,[100,100,1]))
# Residential coefficients output from least squares fit
a_model_slope_res_exp = ls_res.x[0]
b_model_slope_res_exp = ls_res.x[1]
c_model_slope_res_exp = ls_res.x[2]

# Least_squares commercial model, to solve slope = (a + b*(1 - exp(-c*hdd)))/hdd    
ls_com = least_squares(model, [35,35,0.8], args=(np.array(area_scale_com['hdd_normals_2010_popwtd_div1000']), np.array(area_scale_com['htg_slope_com_mmbtu_m2_degC_pophddwtd'])*10**6), method='trf', loss='soft_l1', f_scale=0.1, bounds=(0,[100,100,1]))
# Commercial coefficients output from least squares fit
a_model_slope_com_exp = ls_com.x[0]
b_model_slope_com_exp = ls_com.x[1]
c_model_slope_com_exp = ls_com.x[2]

# Functions with solved coefficients for res and com - produces slopes in btu/m2-C for inputs of HDD
def func_slope_res_exp(x):
    return ((a_model_slope_res_exp + b_model_slope_res_exp*(1 - np.exp(-c_model_slope_res_exp*(x/1000))))/(x/1000))*10**(-6)
def func_slope_com_exp(x):
    return ((a_model_slope_com_exp + b_model_slope_com_exp*(1 - np.exp(-c_model_slope_com_exp*(x/1000))))/(x/1000))*10**(-6)

puma_data['htg_slope_res_mmbtu_m2_degC'] = ''
puma_data['htg_slope_com_mmbtu_m2_degC'] = ''

adj_slopes_res = pd.DataFrame(columns=(['state', 'puma', 'htg_slope_res_mmbtu_m2_degC']))
adj_slopes_com = pd.DataFrame(columns=(['state', 'puma', 'htg_slope_com_mmbtu_m2_degC']))

for state in const.state_list:

    puma_data_it = puma_data[puma_data['state'] == state].reset_index() 
    
    # Residential Adjustments
    # Extract unadjusted heating slope for the given state and class
    htg_slope_res_mmbtu_m2_degC_basis = list(state_slopes_res[state_slopes_res['state'] == state]['sh_slope'])[0]
    # Calculate slope scalar based on hdd, hd, area, and frac_ff
    slope_scalar_res_it = htg_slope_res_mmbtu_m2_degC_basis / (sum(puma_data_it['hdd65_normals_2010'].map(func_slope_res_exp)*puma_data_it['hd_183C_2010']*puma_data_it['res_area_2010_m2']*puma_data_it['frac_ff_sh_res_2010']) / sum(puma_data_it['hd_183C_2010']*puma_data_it['res_area_2010_m2']*puma_data_it['frac_ff_sh_res_2010']))
    # Apply scalar to each puma
    adj_slope_res_list = list(slope_scalar_res_it*puma_data_it['hdd65_normals_2010'].map(func_slope_res_exp))
    for i in range(len(puma_data_it)):
        index_r = len(adj_slopes_res)
        adj_slopes_res.loc[index_r] = [state, puma_data_it['puma'][i], adj_slope_res_list[i]]
    
    # Commercial Adjustments
    htg_slope_com_mmbtu_m2_degC_basis = list(state_slopes_com[state_slopes_com['state'] == state]['sh_slope'])[0]
    slope_scalar_com_it = htg_slope_com_mmbtu_m2_degC_basis / (sum(puma_data_it['hdd65_normals_2010'].map(func_slope_com_exp)*puma_data_it['hd_167C_2010']*puma_data_it['com_area_2010_m2']*puma_data_it['frac_ff_sh_com_2010']) / sum(puma_data_it['hd_167C_2010']*puma_data_it['com_area_2010_m2']*puma_data_it['frac_ff_sh_com_2010']))
    adj_slope_com_list = list(slope_scalar_com_it*puma_data_it['hdd65_normals_2010'].map(func_slope_com_exp))
    for i in range(len(puma_data_it)):
        index_c = len(adj_slopes_com)
        adj_slopes_com.loc[index_c] = [state, puma_data_it['puma'][i], adj_slope_com_list[i]]

# Export climate adjusted space heating slopes of each puma
adj_slopes_res.to_csv(os.path.join(dir_path, "data", "puma_slopes_ff_res.csv"), index = False)
adj_slopes_com.to_csv(os.path.join(dir_path, "data", "puma_slopes_ff_com.csv"), index = False)