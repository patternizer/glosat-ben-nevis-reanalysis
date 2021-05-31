#!/usr/bin/env python

#-----------------------------------------------------------------------
# PROGRAM: glosat-ben-nevis-reanalysis.py
#-----------------------------------------------------------------------
# Version 0.6
# 31 May, 2021
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#-----------------------------------------------------------------------

# Dataframe libraries:
import numpy as np
import pandas as pd
import xarray as xr

# Maths libraries@
import scipy
from sklearn import datasets, linear_model

# Plotting libraries:
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

#-----------------------------------------------------------------------------
# SETTINGS
#-----------------------------------------------------------------------------

fontsize = 16

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def linear_regression_ols(x,y):
    
    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)
    
    X = x.values.reshape(len(x),1)
    t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(t.reshape(-1, 1))
    slope = regr.coef_[0]
    intercept = regr.intercept_
        
#   return t, ypred
    return t, ypred, slope, intercept

#-----------------------------------------------------------------------------
# DATASETS: 1x1
#-----------------------------------------------------------------------------
#land.nc                                 # 1x1 landseamask [1=land]
#-----------------------------------------------------------------------------
# 20CRv3: monthly
#-----------------------------------------------------------------------------
#air.2m.mon.mean.nc                      # 2m-monthly-1836-2015
#air.2m.mon.mean.spread                  # 2m-monthly-1836-2015-spread
#tmax.2m.mon.mean.nc                     # 2m-monthly-1836-2015-tmax
#tmin.2m.mon.mean.nc                     # 2m-monthly-1836-2015-tmin
#air.mon.mean.nc                         # z-monthly-1836-2015
#air.mon.mean.spread                     # z-monthly-1836-2015-spread
#-----------------------------------------------------------------------------
# 20CRv3: 3hourly
#-----------------------------------------------------------------------------
#air_2m_3hourly_1883_1904.nc             # 2m-3hourly-1883-1904
#air_2m_3hourly_spread_1883_1904.nc      # 2m-3hourly-1883-1904-spread
#air_2m_3hourly_BN.nc                    # 2m-3hourly-1883-1904-BN
#air_2m_3hourly_spread_BN.nc             # 2m-3hourly-1883-1904-BN-spread
#air_z_3hourly_BN.nc                     # z-3hourly-1883-1904-BN
#air_z_3hourly_spread_BN.nc              # z-3hourly-1883-1904-BN-spread
#-----------------------------------------------------------------------------
# Observations: (via Ed Hawkins and Stephen Burt, UoR)
#-----------------------------------------------------------------------------
#ben_nevis_summit_daily_v2.csv           # daily-18831201-19040930-Tmin-(oC),Tmax(oC)
#ben_nevis_summit_hourly_v2.csv          # hourly-18831201-19040930-24hr-Tdry-Twet
#fort_william_daily_v2.csv               # daily-18900801-19040930-Tmin-(oC),Tmax(oC)
#fort_william_hourly_v2.csv              # hourly-18900801-19040930-24hr-Tdry-Twet 
#fort_william_school_v2.csv              # hourly-18831201-18901231-5x-Tdry-Twet-Tmin-Tmax

#-----------------------------------------------------------------------------
# LOAD: 20CRv3 (monthly) temperature at 2m and on pressure levels + spread = mean of SDs (1836-2015)
#-----------------------------------------------------------------------------

# Dataset: https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.pressure.html#caveat

ds_20CR_2m = xr.open_dataset('DATA/air.2m.mon.mean.nc', decode_cf=True)
ds_20CR_2m_spread = xr.open_dataset('DATA/air.2m.mon.mean.spread.nc', decode_cf=True)
ds_20CR_2m_tmin = xr.open_dataset('DATA/tmin.2m.mon.mean.nc', decode_cf=True)
ds_20CR_2m_tmax = xr.open_dataset('DATA/tmax.2m.mon.mean.nc', decode_cf=True)
ds_20CR_hPa = xr.open_dataset('DATA/air.mon.mean.nc', decode_cf=True)
ds_20CR_hPa_spread = xr.open_dataset('DATA/air.mon.mean.spread.nc', decode_cf=True)

#-----------------------------------------------------------------------------
# LOAD: 20CRv3 (3-hourly) temperature at 2m and on pressure levels + spread = mean of SDs (1883-1904)
#-----------------------------------------------------------------------------

# Dataset: https://psl.noaa.gov/cgi-bin/db_search/DBSearch.pl?Dataset=NOAA/CIRES/DOE+20th+Century+Reanalysis+version+3si&Variable=Air+temperature&group=0&submit=Search

ds_20CR_2m_3hourly = xr.open_dataset('DATA/air_2m_3hourly_BN.nc', decode_cf=True)
ds_20CR_2m_3hourly_spread = xr.open_dataset('DATA/air_2m_3hourly_spread_BN.nc', decode_cf=True)
ds_20CR_hPa_3hourly = xr.open_dataset('DATA/air_z_3hourly_BN.nc', decode_cf=True)
ds_20CR_hPa_3hourly_spread = xr.open_dataset('DATA/air_z_3hourly_spread_BN.nc', decode_cf=True)

#-----------------------------------------------------------------------------
# LOAD: Ben Nevis (daily tmin,tmax) and hourly tdry,twet) --> Tmean=(Tn+Tx)/2 (monthly) (1883-12-01 to 1904-09-30)
#-----------------------------------------------------------------------------

# Datasets (via Ed Hawkins and Stephen Burt, UoR): https://data.ceda.ac.uk/badc/deposited2019/operation-weather-rescue/data/ben-nevis-data/ben-nevis
 
# HOURLY: Tdry, Twet
f_2m = pd.read_csv('DATA/ben_nevis_summit_hourly_v2.csv', header=4)
fill_Value = -9999.0
f_2m['Temperature (dry bulb, degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m['Temperature (wet bulb, degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m['date'] = [ str(f_2m.iloc[i,0]).zfill(4) + '-' + str(f_2m.iloc[i,1]).zfill(2) + '-' + str(f_2m.iloc[i,2]).zfill(2) + " " + str(f_2m.iloc[i,3]-1).zfill(2) + str(':00:00') for i in range(len(f_2m)) ]
f_2m['date'] = f_2m.loc[:,'date'].astype('datetime64[ns]')
f_2m.set_index('date',inplace=True)

# DAILY: Tmin, Tmax
f_2m_daily = pd.read_csv('DATA/ben_nevis_summit_daily_v2.csv', header=4)
fill_Value = -9999.0
f_2m_daily['Tmin (degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m_daily['Tmax (degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m_daily['Tmean'] = (f_2m_daily['Tmin (degC)']+f_2m_daily['Tmax (degC)'])/2.0 # Tmean = (Tn+Tx)/2 (daily)
f_2m_daily['date'] = [ str(f_2m_daily.iloc[i,0]).zfill(4) + '-' + str(f_2m_daily.iloc[i,1]).zfill(2) + '-' + str(f_2m_daily.iloc[i,2]).zfill(2) + " " + str('09:00:00') for i in range(len(f_2m_daily)) ]
f_2m_daily['date'] = f_2m_daily.loc[:,'date'].astype('datetime64[ns]')
f_2m_daily.set_index('date',inplace=True)

# MERGE: daily and hourly dataframes
f_2m_BN_daily = pd.merge(f_2m, f_2m_daily, left_index=True, right_index=True) # common indices only i.e. at daily time of tmin, tmax
f_2m_BN_hourly = pd.concat([f_2m, f_2m_daily], axis=1) # full merge

#-----------------------------------------------------------------------------
# LOAD: Fort William (daily tmin,tmax) and hourly tdry,twet) --> Tmean=(Tn+Tx)/2 (monthly) (1883-12-01 to 1904-09-30)
#-----------------------------------------------------------------------------

# Datasets (via Ed Hawkins and Stephen Burt, UoR): https://data.ceda.ac.uk/badc/deposited2019/operation-weather-rescue/data/ben-nevis-data/ben-nevis

#-----------------------------------------------------------------------------
# 1) FORT WILLIAM SCHOOL:       1883-12-01 - 1890-12-31
#-----------------------------------------------------------------------------

# HOURLY (5x):  Tdry, Twet
# DAILY:        Tmin, Tmax

#Index(['Year', 'Month', 'Day', 'Sea level pressure (8am)',
#       'Sea level pressure (9am)', 'Sea level pressure (2pm)',
#       'Sea level pressure (6pm)', 'Sea level pressure (9pm)',
#       'Precipitation (9am-9am)', ' Dry Bulb Temperature (8am, degC)',
#       ' Dry Bulb Temperature (9am, degC)',
#       ' Dry Bulb Temperature (2pm, degC)',
#       ' Dry Bulb Temperature (6pm, degC)',
#       ' Dry Bulb Temperature (9pm, degC)',
#       ' Wet Bulb Temperature (8am, degC)',
#       ' Wet Bulb Temperature (9am, degC)',
#       ' Wet Bulb Temperature (2pm, degC)',
#       ' Wet Bulb Temperature (6pm, degC)',
#       ' Wet Bulb Temperature (9pm, degC)', ' Cloud (9am, tenths)',
#       ' Cloud (2pm, tenths)', ' Cloud (9pm, tenths)', ' Tmax (degC)',
#       ' Tmax Observation time', ' Tmin (degC)', ' Tmin Observation time'],      
#      dtype='object')

# Dataframe of 9am observations:
f_2m = pd.read_csv('DATA/fort_william_school_v2.csv', header=4)
f_2m['date'] = [ str(f_2m.iloc[i,0]).zfill(4) + '-' + str(f_2m.iloc[i,1]).zfill(2) + '-' + str(f_2m.iloc[i,2]).zfill(2) + " " + str('09:00:00') for i in range(len(f_2m)) ]
f_2m['date'] = f_2m.loc[:,'date'].astype('datetime64[ns]')
f_2m.set_index('date',inplace=True)
f_2m['Tmean'] = (f_2m[' Tmin (degC)']+f_2m[' Tmax (degC)'])/2.0 # Tmean = (Tn+Tx)/2 (daily)

f_2m_FW1_daily = f_2m.copy(); 
f_2m_FW1_daily.drop([
        'Sea level pressure (8am)',
        'Sea level pressure (2pm)',
        'Sea level pressure (6pm)', 
        'Sea level pressure (9pm)',
        ' Dry Bulb Temperature (8am, degC)',
        ' Dry Bulb Temperature (2pm, degC)',
        ' Dry Bulb Temperature (6pm, degC)',
        ' Dry Bulb Temperature (9pm, degC)',
        ' Wet Bulb Temperature (8am, degC)',    
        ' Wet Bulb Temperature (2pm, degC)',    
        ' Wet Bulb Temperature (6pm, degC)',
        ' Wet Bulb Temperature (9pm, degC)', 
        ' Cloud (9am, tenths)', ' Cloud (2pm, tenths)', ' Cloud (9pm, tenths)',        
        ' Tmax Observation time', ' Tmin Observation time'       
        ], axis=1, inplace=True)
f_2m_FW1_daily.rename({
        'Sea level pressure (9am)':'Sea level pressure (mb)',
        ' Dry Bulb Temperature (9am, degC)':'Temperature (dry bulb, degC)',
        ' Wet Bulb Temperature (9am, degC)':'Temperature (wet bulb, degC)',
        'Precipitation (9am-9am)':'Precipitation (mm)',
        ' Tmin (degC)':'Tmin (degC)',
        ' Tmax (degC)':'Tmax (degC)'                       
        }, axis=1, inplace=True)
f_2m_FW1_daily_reordered = f_2m_FW1_daily[[  'Year', 'Month', 'Day', 'Sea level pressure (mb)',
       'Temperature (dry bulb, degC)', 'Temperature (wet bulb, degC)',
       'Precipitation (mm)', 'Tmin (degC)', 'Tmax (degC)', 'Tmean' ]]

# Dataframe of 8am observations:
g_8am = f_2m_FW1_daily_reordered.copy(); g_8am.iloc[:,3:] = np.nan; g_8am.index = f_2m_FW1_daily_reordered.index + pd.DateOffset(hours=-1)
g_8am['Temperature (dry bulb, degC)'] = f_2m[' Dry Bulb Temperature (8am, degC)'].values
g_8am['Temperature (wet bulb, degC)'] = f_2m[' Wet Bulb Temperature (8am, degC)'].values

# Dataframe of 2pm observations:
g_2pm = f_2m_FW1_daily_reordered.copy(); g_2pm.iloc[:,3:] = np.nan; g_2pm.index = f_2m_FW1_daily_reordered.index + pd.DateOffset(hours=5)
g_2pm['Temperature (dry bulb, degC)'] = f_2m[' Dry Bulb Temperature (2pm, degC)'].values
g_2pm['Temperature (wet bulb, degC)'] = f_2m[' Wet Bulb Temperature (2pm, degC)'].values

# Dataframe of 6pm observations:
g_6pm = f_2m_FW1_daily_reordered.copy(); g_6pm.iloc[:,3:] = np.nan; g_6pm.index = f_2m_FW1_daily_reordered.index + pd.DateOffset(hours=9)
g_6pm['Temperature (dry bulb, degC)'] = f_2m[' Dry Bulb Temperature (6pm, degC)'].values
g_6pm['Temperature (wet bulb, degC)'] = f_2m[' Wet Bulb Temperature (6pm, degC)'].values

# Dataframe of 9pm observations:
g_9pm = f_2m_FW1_daily_reordered.copy(); g_9pm.iloc[:,3:] = np.nan; g_9pm.index = f_2m_FW1_daily_reordered.index + pd.DateOffset(hours=12)
g_9pm['Temperature (dry bulb, degC)'] = f_2m[' Dry Bulb Temperature (9pm, degC)'].values
g_9pm['Temperature (wet bulb, degC)'] = f_2m[' Wet Bulb Temperature (9pm, degC)'].values

f_2m_FW1_8am = pd.concat([f_2m_FW1_daily_reordered, g_8am], axis=0)
f_2m_FW1_2pm = pd.concat([f_2m_FW1_8am, g_2pm], axis=0)
f_2m_FW1_6pm = pd.concat([f_2m_FW1_2pm, g_6pm], axis=0)
f_2m_FW1_9pm = pd.concat([f_2m_FW1_6pm, g_9pm], axis=0)
f_2m_FW1_hourly_reordered = f_2m_FW1_9pm.sort_index()

# TRIM: to 1890-07-31 (because FW2 starts on 1890-08-01)
f_2m_FW1_hourly_reordered_trimmed = f_2m_FW1_hourly_reordered[f_2m_FW1_hourly_reordered.index<'1890-08-01']
f_2m_FW1_daily_reordered_trimmed = f_2m_FW1_daily_reordered[f_2m_FW1_daily_reordered.index<'1890-08-01']

#-----------------------------------------------------------------------------
# 2) FORT WILLIAM OBSERVATORY:  1890-08-01 - 1904-09-30
#-----------------------------------------------------------------------------

# HOURLY: Tdry, Twet
f_2m = pd.read_csv('DATA/fort_william_hourly_v2.csv', header=4)
fill_Value = -9999.0
f_2m['Temperature (dry bulb, degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m['Temperature (wet bulb, degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m['date'] = [ str(f_2m.iloc[i,0]).zfill(4) + '-' + str(f_2m.iloc[i,1]).zfill(2) + '-' + str(f_2m.iloc[i,2]).zfill(2) + " " + str(f_2m.iloc[i,3]-1).zfill(2) + str(':00:00') for i in range(len(f_2m)) ]
f_2m['date'] = f_2m.loc[:,'date'].astype('datetime64[ns]')
f_2m.set_index('date',inplace=True)

# DAILY: Tmin, Tmax
f_2m_daily = pd.read_csv('DATA/fort_william_daily_v2.csv', header=4)
fill_Value = -9999.0
f_2m_daily['Tmin (degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m_daily['Tmax (degC)'].replace(fill_Value,np.nan, inplace=True)
f_2m_daily['Tmean'] = (f_2m_daily['Tmin (degC)']+f_2m_daily['Tmax (degC)'])/2.0 # Tmean = (Tn+Tx)/2 (daily)
f_2m_daily['date'] = [ str(f_2m_daily.iloc[i,0]).zfill(4) + '-' + str(f_2m_daily.iloc[i,1]).zfill(2) + '-' + str(f_2m_daily.iloc[i,2]).zfill(2) + " " + str('09:00:00') for i in range(len(f_2m_daily)) ]
f_2m_daily['date'] = f_2m_daily.loc[:,'date'].astype('datetime64[ns]')
f_2m_daily.set_index('date',inplace=True)

# MERGE: daily Tmin,Tmax into Tdry,Twet dataframe and extract at common timestamp (9am)
f_2m_FW2_daily = pd.merge(f_2m, f_2m_daily.iloc[:,3:7], left_index=True, right_index=True) # common indices only i.e. at daily time of tmin, tmax
f_2m_FW2_daily.drop(['Hour','Precipitation (mm)_x'], axis=1, inplace=True)
f_2m_FW2_daily.rename({'Precipitation (mm)_y':'Precipitation (mm)'}, axis=1, inplace=True)

# MERGE: daily Tmin,Tmax into hourly Tdry,Twet dataframe
f_2m.drop(['Hour'], axis=1, inplace=True)
f_2m_daily.drop(['Year','Month','Day','Precipitation (mm)'], axis=1, inplace=True)
f_2m_FW2_hourly = pd.concat([f_2m, f_2m_daily], axis=1)

f_2m_FW2_hourly_reordered = f_2m_FW2_hourly[[  'Year', 'Month', 'Day', 'Sea level pressure (mb)',
       'Temperature (dry bulb, degC)', 'Temperature (wet bulb, degC)',
       'Precipitation (mm)', 'Tmin (degC)', 'Tmax (degC)', 'Tmean' ]]

# MERGE: both stations
f_2m_FW_daily = pd.concat([f_2m_FW1_daily_reordered_trimmed, f_2m_FW2_daily])
f_2m_FW_hourly = pd.concat([f_2m_FW1_hourly_reordered_trimmed, f_2m_FW2_hourly_reordered])

#-----------------------------------------------------------------------------
# EXTRACT: reanalysis at 850hPa level and convert to degC
#-----------------------------------------------------------------------------

dr_20CR_2m_3hourly = ds_20CR_2m_3hourly.air[:,0,0]-273.15
dr_20CR_2m_3hourly_spread = ds_20CR_2m_3hourly_spread.air[:,0,0]
dr_20CR_hPa_3hourly = ds_20CR_hPa_3hourly.air[:,5,0,0]-273.15
dr_20CR_hPa_3hourly_spread = ds_20CR_hPa_3hourly_spread.air[:,5,0,0]

# CONSTRUCT: dataframe with datetime index

t = dr_20CR_2m_3hourly.time
T_20CRv3 = pd.DataFrame({
        'T(850hPa)':dr_20CR_hPa_3hourly.values, 
        'T(2m)':dr_20CR_2m_3hourly.values, 
        'T(850hPa) spread':dr_20CR_hPa_3hourly_spread.values, 
        'T(2m) spread':dr_20CR_2m_3hourly_spread.values})                             
T_20CRv3['date'] = [ str(t[i].values)[0:4] + '-' + str(t[i].values)[4:6] + '-' + str(t[i].values)[6:8] + ' ' + str(int( (t[i]-np.floor(t[i])).values*24 )).zfill(2) + ':00:00' for i in range(len(t)) ]
T_20CRv3['date'] = T_20CRv3.loc[:,'date'].astype('datetime64[ns]')
T_20CRv3.set_index('date',inplace=True)

#-----------------------------------------------------------------------------
# MERGE: reanalysis and observations
#-----------------------------------------------------------------------------

FW = T_20CRv3.merge(f_2m_FW_hourly, left_index=True, right_index=True)
BN = T_20CRv3.merge(f_2m_BN_hourly, left_index=True, right_index=True)

#-----------------------------------------------------------------------------
# CALCULATE: correlation between reanalysis and observations
#-----------------------------------------------------------------------------

print('plotting Fort William correlations ...')
    
for hh in range(0,24,3):
    
    X = FW[FW.index.hour==hh]['T(2m)']
    Y_dry = FW[FW.index.hour==hh]['Temperature (dry bulb, degC)'].astype(float)
    Y_wet = FW[FW.index.hour==hh]['Temperature (wet bulb, degC)'].astype(float)
    Y_min = FW[FW.index.hour==hh]['Tmin (degC)']
    Y_max = FW[FW.index.hour==hh]['Tmax (degC)']
    minval = np.nanmin([np.nanmin(X),np.nanmin(Y_dry),np.nanmin(Y_wet),np.nanmin(Y_max),np.nanmin(Y_min)])
    maxval = np.nanmax([np.nanmax(X),np.nanmax(Y_dry),np.nanmax(Y_wet),np.nanmax(Y_max),np.nanmax(Y_min)])

    #------------------------------------------------------------------------------
    # OLS: linear regression
    #------------------------------------------------------------------------------
        
    mask_dry = np.isfinite(X) & np.isfinite(Y_dry)
    mask_wet = np.isfinite(X) & np.isfinite(Y_wet)
    corrcoef_dry = scipy.stats.pearsonr(X[mask_dry], Y_dry[mask_dry])[0]
    corrcoef_wet = scipy.stats.pearsonr(X[mask_wet], Y_wet[mask_wet])[0]
    OLS_X_dry, OLS_Y_dry, OLS_slope_dry, OLS_intercept_dry = linear_regression_ols(X[mask_dry], Y_dry[mask_dry])
    OLS_X_wet, OLS_Y_wet, OLS_slope_wet, OLS_intercept_wet = linear_regression_ols(X[mask_wet], Y_wet[mask_wet])
   
    figstr = 'fort-william-correlations-' + str(hh).zfill(2) + '00' + '.png'
    titlestr = 'Fort William: observations versus T(2m) reanalysis (' + str(hh).zfill(2) + ':00)'
                              
    fig,ax = plt.subplots(figsize=(15,10))    
    plt.scatter(X, Y_dry, alpha=0.1, marker='o', color='violet', s=5, facecolor='pink', ls='-', lw=1)
    plt.scatter(X, Y_wet, alpha=0.1, marker='o', color='turquoise', s=5, facecolor='cyan', ls='-', lw=1)
    if hh == 9:
        mask_min = np.isfinite(X) & np.isfinite(Y_min)
        mask_max = np.isfinite(X) & np.isfinite(Y_max)
        corrcoef_min = scipy.stats.pearsonr(X[mask_min], Y_min[mask_min])[0]
        corrcoef_max = scipy.stats.pearsonr(X[mask_max], Y_max[mask_max])[0]    
        OLS_X_min, OLS_Y_min, OLS_slope_min, OLS_intercept_min = linear_regression_ols(X[mask_min], Y_min[mask_min])
        OLS_X_max, OLS_Y_max, OLS_slope_max, OLS_intercept_max = linear_regression_ols(X[mask_max], Y_max[mask_max])
        plt.scatter(X, Y_min, alpha=0.1, marker='o', color='blue', s=5, facecolor=None, ls='-', lw=1, zorder=0)
        plt.scatter(X, Y_max, alpha=0.1, marker='o', color='red', s=5, facecolor=None, ls='-', lw=1, zorder=0)
        plt.plot(OLS_X_min, OLS_Y_min, color='blue', ls='-', lw=2, label=r'T(min): OLS $\rho$='+str(np.round(corrcoef_min,3)) + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_min,3)))
        plt.plot(OLS_X_max, OLS_Y_max, color='red', ls='-', lw=2, label=r'T(max): OLS $\rho$='+str(np.round(corrcoef_max,3)) + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_max,3)))

    plt.plot(OLS_X_dry, OLS_Y_dry, color='violet', ls='-', lw=2, label=r'T(dry bulb): OLS $\rho$='+str(np.round(corrcoef_dry,3))  + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_dry,3)))
    plt.plot(OLS_X_wet, OLS_Y_wet, color='turquoise', ls='-', lw=2, label=r'T(wet bulb): OLS $\rho$='+str(np.round(corrcoef_wet,3))  + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_wet,3)))
    ax.plot([minval,maxval], [minval,maxval], color='black', ls='--', zorder=10)    
    ax.set_xlim(minval, maxval)
    ax.set_ylim(minval, maxval)
    ax.set_aspect('equal') 
    ax.xaxis.grid(True, which='minor')      
    ax.yaxis.grid(True, which='minor')  
    ax.xaxis.grid(True, which='major')      
    ax.yaxis.grid(True, which='major')  
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    leg = plt.legend(loc='lower right', ncol=1, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    for l in leg.get_lines(): 
        l.set_alpha(1)
        l.set_marker('.')
    plt.xlabel("20CRv3 absolute temperature (2m), $\mathrm{\degree}C$", fontsize=fontsize)
    plt.ylabel("Observed absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

print('plotting Ben Nevis correlations ...')
    
for hh in range(0,24,3):
    
    X = BN[BN.index.hour==hh]['T(850hPa)']
    Y_dry = BN[BN.index.hour==hh]['Temperature (dry bulb, degC)'].astype(float)
    Y_wet = BN[BN.index.hour==hh]['Temperature (wet bulb, degC)'].astype(float)
    Y_min = BN[BN.index.hour==hh]['Tmin (degC)']
    Y_max = BN[BN.index.hour==hh]['Tmax (degC)']
    minval = np.nanmin([np.nanmin(X),np.nanmin(Y_dry),np.nanmin(Y_wet),np.nanmin(Y_max),np.nanmin(Y_min)])
    maxval = np.nanmax([np.nanmax(X),np.nanmax(Y_dry),np.nanmax(Y_wet),np.nanmax(Y_max),np.nanmax(Y_min)])

    #------------------------------------------------------------------------------
    # OLS: linear regression
    #------------------------------------------------------------------------------
        
    mask_dry = np.isfinite(X) & np.isfinite(Y_dry)
    mask_wet = np.isfinite(X) & np.isfinite(Y_wet)
    corrcoef_dry = scipy.stats.pearsonr(X[mask_dry], Y_dry[mask_dry])[0]
    corrcoef_wet = scipy.stats.pearsonr(X[mask_wet], Y_wet[mask_wet])[0]
    OLS_X_dry, OLS_Y_dry, OLS_slope_dry, OLS_intercept_dry = linear_regression_ols(X[mask_dry], Y_dry[mask_dry])
    OLS_X_wet, OLS_Y_wet, OLS_slope_wet, OLS_intercept_wet = linear_regression_ols(X[mask_wet], Y_wet[mask_wet])
   
    figstr = 'ben-nevis-correlations-' + str(hh).zfill(2) + '00' + '.png'
    titlestr = 'Ben Nevis: observations versus T(850hPa) reanalysis (' + str(hh).zfill(2) + ':00)'
                              
    fig,ax = plt.subplots(figsize=(15,10))    
    plt.scatter(X, Y_dry, alpha=0.1, marker='o', color='violet', s=5, facecolor='pink', ls='-', lw=1)
    plt.scatter(X, Y_wet, alpha=0.1, marker='o', color='turquoise', s=5, facecolor='cyan', ls='-', lw=1)
    if hh == 9:
        mask_min = np.isfinite(X) & np.isfinite(Y_min)
        mask_max = np.isfinite(X) & np.isfinite(Y_max)
        corrcoef_min = scipy.stats.pearsonr(X[mask_min], Y_min[mask_min])[0]
        corrcoef_max = scipy.stats.pearsonr(X[mask_max], Y_max[mask_max])[0]    
        OLS_X_min, OLS_Y_min, OLS_slope_min, OLS_intercept_min = linear_regression_ols(X[mask_min], Y_min[mask_min])
        OLS_X_max, OLS_Y_max, OLS_slope_max, OLS_intercept_max = linear_regression_ols(X[mask_max], Y_max[mask_max])
        plt.scatter(X, Y_min, alpha=0.1, marker='o', color='blue', s=5, facecolor=None, ls='-', lw=1, zorder=0)
        plt.scatter(X, Y_max, alpha=0.1, marker='o', color='red', s=5, facecolor=None, ls='-', lw=1, zorder=0)
        plt.plot(OLS_X_min, OLS_Y_min, color='blue', ls='-', lw=2, label=r'T(min): OLS $\rho$='+str(np.round(corrcoef_min,3)) + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_min,3)))
        plt.plot(OLS_X_max, OLS_Y_max, color='red', ls='-', lw=2, label=r'T(max): OLS $\rho$='+str(np.round(corrcoef_max,3)) + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_max,3)))

    plt.plot(OLS_X_dry, OLS_Y_dry, color='violet', ls='-', lw=2, label=r'T(dry bulb): OLS $\rho$='+str(np.round(corrcoef_dry,3))  + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_dry,3)))
    plt.plot(OLS_X_wet, OLS_Y_wet, color='turquoise', ls='-', lw=2, label=r'T(wet bulb): OLS $\rho$='+str(np.round(corrcoef_wet,3))  + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_wet,3)))
    ax.plot([minval,maxval], [minval,maxval], color='black', ls='--', zorder=10)    
    ax.set_xlim(minval, maxval)
    ax.set_ylim(minval, maxval)
    ax.set_aspect('equal') 
    ax.xaxis.grid(True, which='minor')      
    ax.yaxis.grid(True, which='minor')  
    ax.xaxis.grid(True, which='major')      
    ax.yaxis.grid(True, which='major')  
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    leg = plt.legend(loc='lower right', ncol=1, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    for l in leg.get_lines(): 
        l.set_alpha(1)
        l.set_marker('.')
    plt.xlabel("20CRv3 absolute temperature (850hPa), $\mathrm{\degree}C$", fontsize=fontsize)
    plt.ylabel("Observed absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
        
print('plotting Ben Nevis 3-hourly ridgelines ...')

def three_hourly_ridgelines(place, var, df):
    
    df['Hour'] = df.index.hour
    hour_dict = {0: '00:00', 3: '03:00', 6: '06:00', 9: '09:00', 12: '12:00', 15: '15:00', 18: '18:00', 21: '21:00'}    
    df['Hour'] = df['Hour'].map(hour_dict)
    df_hour = df.groupby('Hour')[var].mean()
    df['mean_Hour'] = df['Hour'].map(df_hour)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 15)
    palette = sns.color_palette(palette='coolwarm', n_colors=10)
    g = sns.FacetGrid(df, row='Hour', hue='mean_Hour', aspect=20, height=0.75, palette=palette)
    g.map(sns.kdeplot, var, bw_adjust=1, clip_on=False, fill=True, alpha=1, linewidth=1.5) 
    g.map(sns.kdeplot, var, bw_adjust=1, clip_on=False, color="w", lw=2) # add a white line for contour of each kdeplot
    g.map(plt.axhline, y=0, lw=2, clip_on=False) # add horizontal line for each plot
    # loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
    # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
    for i, ax in enumerate(g.axes.flat):
        ax.text(-20, 0.02, hour_dict[3*i], color=ax.lines[-1].get_color(), fontweight='bold', fontsize=fontsize)    
    g.fig.subplots_adjust(hspace=-0.3)
    g.set_titles("")
    g.set(yticks=[])
    g.set(xlim=[-20,20])
    g.despine(bottom=True, left=True)
    g.fig.suptitle(place + ': ' + var + ' per 3-hour interval: 1883-1904', ha='right', fontsize=fontsize, fontweight=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize, fontweight='bold')
    #plt.xlabel('Absoute temperature $\mathrm{\degree}C$', fontweight='bold', fontsize=fontsize)
    #fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)             
    plt.savefig(place + '-' + var + '-ridgeline.png', dpi=300)
    plt.close('all')

three_hourly_ridgelines('Ben Nevis', 'Temperature (dry bulb, degC)', BN)
three_hourly_ridgelines('Ben Nevis', 'Temperature (wet bulb, degC)', BN)

#-----------------------------------------------------------------------------
# FUNCTION: to extract and plot events (observations versus reanalysis)
#-----------------------------------------------------------------------------
    
def event_extract_and_plot(event_label, event_timestamp, event_startdate, event_enddate):

    #-----------------------------------------------------------------------------
    # SLICE: by event startdate and enddate
    #-----------------------------------------------------------------------------

    event_startdate_decimal = float(int((event_startdate.replace('-', ''))))
    event_enddate_decimal = float(int((event_enddate.replace('-', ''))))
        
    f_2m_BN_hourly_event = f_2m_BN_hourly[ (f_2m_BN_hourly.index >= event_startdate) & (f_2m_BN_hourly.index <= event_enddate) ]
    f_2m_FW_hourly_event = f_2m_FW_hourly[ (f_2m_FW_hourly.index >= event_startdate) & (f_2m_FW_hourly.index <= event_enddate) ]
    f_2m_BN_daily_event = f_2m_BN_daily[ (f_2m_BN_daily.index >= event_startdate) & (f_2m_BN_daily.index <= event_enddate) ]
    f_2m_FW_daily_event = f_2m_FW_daily[ (f_2m_FW_daily.index >= event_startdate) & (f_2m_FW_daily.index <= event_enddate) ]
    
    dr_20CR_2m_3hourly_event = dr_20CR_2m_3hourly.sel(time=slice(event_startdate_decimal, event_enddate_decimal))
    dr_20CR_2m_3hourly_spread_event = dr_20CR_2m_3hourly_spread.sel(time=slice(event_startdate_decimal, event_enddate_decimal))    
    dr_20CR_hPa_3hourly_event = dr_20CR_hPa_3hourly.sel(time=slice(event_startdate_decimal, event_enddate_decimal))
    dr_20CR_hPa_3hourly_spread_event = dr_20CR_hPa_3hourly_spread.sel(time=slice(event_startdate_decimal, event_enddate_decimal))
    t = dr_20CR_hPa_3hourly_event.time
    f_20CRv3_3hourly_event = pd.DataFrame({
        'T(850hPa)':dr_20CR_hPa_3hourly_event.values, 
        'T(2m)':dr_20CR_2m_3hourly_event.values, 
        'T(850hPa) spread':dr_20CR_hPa_3hourly_spread_event.values, 
        'T(2m) spread':dr_20CR_2m_3hourly_spread_event.values})                             
    f_20CRv3_3hourly_event['date'] = [ str(t[i].values)[0:4] + '-' + str(t[i].values)[4:6] + '-' + str(t[i].values)[6:8] + ' ' + str(int( (t[i]-np.floor(t[i])).values*24 )).zfill(2) + ':00:00' for i in range(len(dr_20CR_hPa_3hourly_event)) ]
    f_20CRv3_3hourly_event['date'] = f_20CRv3_3hourly_event.loc[:,'date'].astype('datetime64[ns]')
    f_20CRv3_3hourly_event.set_index('date',inplace=True)
    
    #-----------------------------------------------------------------------------
    # PLOT: sliced observations versus reanalysis
    #-----------------------------------------------------------------------------
    
    print('plotting Ben Nevis ...')
    
    figstr = 'ben-nevis-850hPa-observations-' + event_startdate.replace('-','') + '-' + event_enddate.replace('-','') + '.png'
    titlestr = 'Ben Nevis: observations (' + event_startdate + ' to ' + event_enddate +')'
    
    fig,ax = plt.subplots(figsize=(15,10))
    plt.plot(f_2m_BN_hourly_event.index, f_2m_BN_hourly_event['Temperature (dry bulb, degC)'], 'o', color='violet',  ls='--', markersize=5, markerfacecolor='pink', markeredgewidth=1, alpha=0.75, zorder=1, label='Ben Nevis: hourly T(dry bulb)')
    plt.plot(f_2m_BN_hourly_event.index, f_2m_BN_hourly_event['Temperature (wet bulb, degC)'], 'o', color='turquoise', lw=1, ls='--', markersize=5, markerfacecolor='cyan', markeredgewidth=1, alpha=0.75, zorder=1, label='Ben Nevis: hourly T(wet bulb)')
    plt.plot(f_2m_BN_hourly_event.index, f_2m_BN_hourly_event['Tmax (degC)'], 'o', color='red', lw=3, markersize=5, markerfacecolor='white', markeredgewidth=1, zorder=10, label='Ben Nevis: daily Tmax')
    #plt.plot(f_2m_BN_hourly_event.index, f_2m_BN_hourly_event['Tmean'], 'o', color='teal',  lw=3, markersize=5, markerfacecolor='white', markeredgewidth=1, zorder=30, label=None)    
    plt.plot(f_2m_BN_hourly_event.index, f_2m_BN_hourly_event['Tmin (degC)'], 'o', color='blue', lw=3, markersize=5, markerfacecolor='white', markeredgewidth=1, zorder=10, label='Ben Nevis: daily Tmin')
    plt.plot(f_2m_BN_daily_event.index, f_2m_BN_daily_event['Tmean'], color='teal', lw=3, zorder=20, label='Ben Nevis: daily Tmean=(Tn+Tx)/2')    
    plt.axvline(x=pd.Timestamp(event_timestamp), color='black', lw=1, ls='--', label=event_label)
    plt.plot(f_20CRv3_3hourly_event.index, f_20CRv3_3hourly_event['T(850hPa)'].values, color='purple', alpha=0.2, lw=1, zorder=0)
    plt.plot(f_20CRv3_3hourly_event.index, pd.Series(f_20CRv3_3hourly_event['T(850hPa)'].values).rolling(window=8, center=True).mean(), color='purple', alpha=1.0, lw=3, zorder=5, label='20CRv3: 3-hourly T(850hPa) 1d MA') # 1888-01 - 1903-12 at Ben Nevis [-5.55] on 1 degree grid
    plt.fill_between(f_20CRv3_3hourly_event.index, pd.Series(f_20CRv3_3hourly_event['T(850hPa)'].values).rolling(window=8, center=True).mean()-pd.Series(f_20CRv3_3hourly_event['T(850hPa) spread'].values).rolling(window=8, center=True).mean(), pd.Series(f_20CRv3_3hourly_event['T(850hPa)'].values).rolling(window=8, center=True).mean()+pd.Series(f_20CRv3_3hourly_event['T(850hPa) spread'].values).rolling(window=8, center=True).mean(), color='purple', alpha=0.1, zorder=0, label='20CRv3: 3-hourly T(850hPa) 1d MA $\pm$ ens SD') 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.tick_params(labelsize=fontsize)
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.8), ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)             
    plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    print('plotting Fort William ...')
    
    figstr = 'fort-william-2m-observations-' + event_startdate.replace('-','') + '-' + event_enddate.replace('-','') + '.png'
    titlestr = 'Fort William: observations (' + event_startdate + ' to ' + event_enddate +')'
    
    fig,ax = plt.subplots(figsize=(15,10))
    plt.plot(f_2m_FW_hourly_event.index, f_2m_FW_hourly_event['Temperature (dry bulb, degC)'], 'o', color='violet',  ls='--', markersize=5, markerfacecolor='pink', markeredgewidth=1, alpha=0.75, zorder=1, label='Fort William: hourly T(dry bulb)')
    plt.plot(f_2m_FW_hourly_event.index, f_2m_FW_hourly_event['Temperature (wet bulb, degC)'], 'o', color='turquoise', lw=1, ls='--', markersize=5, markerfacecolor='cyan', markeredgewidth=1, alpha=0.75, zorder=1, label='Fort William: hourly T(wet bulb)')
    plt.plot(f_2m_FW_hourly_event.index, f_2m_FW_hourly_event['Tmax (degC)'], 'o', color='red', lw=3, markersize=5, markerfacecolor='white', markeredgewidth=1, zorder=10, label='Fort William: daily Tmax')
    #plt.plot(f_2m_FW_hourly_event.index, f_2m_FW_hourly_event['Tmean'], 'o', color='teal',  lw=3, markersize=5, markerfacecolor='white', markeredgewidth=1, zorder=30, label=None)    
    plt.plot(f_2m_FW_hourly_event.index, f_2m_FW_hourly_event['Tmin (degC)'], 'o', color='blue', lw=3, markersize=5, markerfacecolor='white', markeredgewidth=1, zorder=10, label='Fort William: daily Tmin')
    plt.plot(f_2m_FW_daily_event.index, f_2m_FW_daily_event['Tmean'], color='teal', lw=3, zorder=20, label='Fort William: daily Tmean=(Tn+Tx)/2')    
    plt.axvline(x=pd.Timestamp(event_timestamp), color='black', lw=1, ls='--', label=event_label)
    plt.plot(f_20CRv3_3hourly_event.index, f_20CRv3_3hourly_event['T(2m)'].values, color='purple', alpha=0.2, lw=1, zorder=0)
    plt.plot(f_20CRv3_3hourly_event.index, pd.Series(f_20CRv3_3hourly_event['T(2m)'].values).rolling(window=8, center=True).mean(), color='purple', alpha=1.0, lw=3, zorder=5, label='20CRv3: 3-hourly T(2m) 1d MA') # 1888-01 - 1903-12 at Ben Nevis [-5.55] on 1 degree grid
    plt.fill_between(f_20CRv3_3hourly_event.index, pd.Series(f_20CRv3_3hourly_event['T(2m)'].values).rolling(window=8, center=True).mean()-pd.Series(f_20CRv3_3hourly_event['T(2m) spread'].values).rolling(window=8, center=True).mean(), pd.Series(f_20CRv3_3hourly_event['T(2m)'].values).rolling(window=8, center=True).mean()+pd.Series(f_20CRv3_3hourly_event['T(2m) spread'].values).rolling(window=8, center=True).mean(), color='purple', alpha=0.1, zorder=0, label='20CRv3: 3-hourly T(2m) 1d MA $\pm$ ens SD') 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.tick_params(labelsize=fontsize)
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.8), ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)             
    plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#event_label = 'Ulysses Storm: 1903-02-27'
#event_timestamp = '1903-02-27 00:00:00'
#event_startdate = '1903-02-01'
#event_enddate = '1903-04-01'
#event_extract_and_plot(event_label, event_timestamp, event_startdate, event_enddate)
    
event_extract_and_plot('Storm (p115): 1884-01-21', '1884-01-21 00:00:00', '1883-12-01', '1884-02-28')
event_extract_and_plot('Storm (p116): 1885-01-11', '1885-01-11 00:00:00', '1884-12-01', '1885-02-28')
event_extract_and_plot('Storm (p116): 1885-02-21', '1885-02-21 00:00:00', '1885-01-01', '1885-03-31')
event_extract_and_plot('Heavy Rain (p88): 1890-10', '1890-10-27 00:00:00', '1890-10-01', '1890-10-31')
event_extract_and_plot('Storm (p120): 1891-01-31', '1891-01-31 00:00:00', '1891-01-01', '1891-02-28')
event_extract_and_plot('Storm (p82): 1898-11-22', '1898-11-22 00:00:00', '1898-11-01', '1898-12-31')
event_extract_and_plot('Storm (p123): 1900-06-12', '1900-06-12 00:00:00', '1900-05-01', '1900-07-31')
event_extract_and_plot('Ulysses Storm: 1903-02-27', '1903-02-27 00:00:00', '1903-02-01', '1903-04-01')


#-----------------------------------------------------------------------------
print('** END')
