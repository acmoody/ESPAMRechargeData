# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:42:20 2018

@author: amoody
"""
import os
import pandas as pd
import numpy as np
import re
import subprocess as sub
import matplotlib.pyplot as plt
plt.style.use('bmh')
# ETIdaho directory for Recharge 
ETIdir = 'D:/ESRP/RechargeData_Alex/FPT/Wetlands/ETIdaho/'

def parse_ETIDAHO_monthly( ETIdir ,  cropre = None):
    
    # Get list of Sites (this is a bytestring list)
    files = sub.check_output(['ls',ETIdir + '*_monthly.dat']).split()

    d = {}
    #sitenames = {}
    crops = {}
    #df3=pd.DataFrame()
    for site in files:   
        # Extract Site name  and crops
        with open(site) as infile:
            print('Parsing {}...'.format(site))
            flines = infile.readlines()
            infile.close()
            namestr = re.findall(r'Results for (.*?) .Computed',flines[0])[0]
            # Capitalize
            namestr = namestr.upper()
            # Remove punctuation and white spaces
            namestr = namestr.replace(' ','_')
            namestr = namestr.replace('.','')
            #sitenames[site] = namestr
            codestr = re.findall(r'(\d{6,})ETc',site.decode())[0]

        
            ## Crops ------
            # Original method that messes up crop codes
            # 2 Regex
            a= re.findall(r'(\d.*)\d*',flines[8])  
            b = re.split(r'\d{1,2}', ''.join(a))
            b = [x.strip(' ') for x in b]
              # Conversion to string adds an empty '' row, remove      
            cropnames = filter(None,b)
            cropnames = b
            crops[site]= cropnames
            # get crop codes
            cropcodes =  re.findall(r'(\d{1,2})' ,flines[8].strip(' '))           
        
        # Get ET data    
        df = pd.read_csv(site.decode(),
                        sep='\s+',
                        skipinitialspace=True,
                        skiprows = range(0,9),
                        na_values = [-999],
                        parse_dates = { 'Date': [0, 1]},
                        infer_datetime_format = True,
                        index_col = 'Date')
        
        #mask = (df.index >= '1985-01-01') & (df.index <= '2015-12-31')
        # or df.loc['2000-1-1':'2015-1-1']
        #df = df.loc[mask]
        # There are some weird values the columns. Check them to see if the separator
        # was applied correctly. Fix for now.
        df = df.apply(pd.to_numeric, errors= 'coerce')
        # Extract V.Dys and ETact columns for all crop types 
        # Convert mm/day to mm/month
        ET = df.filter(regex = 'ETact')
        #days = df.filter(regex = 'V.Dys')
        df2 = ET.apply(lambda x: x.index.daysinmonth * x)
        #df3[namestr] = df2
        # Label with crop codes
        df2.columns = cropnames[1:]
      
        # Filter out seleceted crops if cropnames or cropcodes exist
        if cropre:
            df2 = df2.filter(regex=cropre)
        
        # Place station in dictionary
        d[tuple([namestr,codestr])] = df2
        
    return d
    

d = parse_ETIDAHO_monthly(ETIdir,cropre = r'Wetland')
a=pd.concat(d,axis=0)
b=a.unstack(level=[0,1])
b.sort_index(level=1,axis=1,inplace=True)
b.columns = b.columns.swaplevel(1,0)

# --------- 
# Longest consecutive period
# -------------
def getLongestRec( s ):
    
    x = pd.Series(
            ( np.diff(s.dropna().index.to_julian_date()) > 93).astype(int).cumsum(),name='Y')
    groups = x.reset_index().groupby('Y')['index'].apply(np.array)
    groups_len = groups.apply(lambda x: len(x))
    groups_long_idx = groups_len.values.argmax()
    lidx = groups[groups_long_idx] 
    
    return s.dropna()[lidx+1].resample('MS').asfreq()
# ------------
# ARIMA!!!
# -----------------
import statsmodels.api as sm
import itertools
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
# GRID SEARCH PARAMS
# Make tuples of ARIMA order to search
p = q = range(0, 3)
d = range(0,2)
pdq = list(itertools.product(p, d, q))
#seasonal_pdq = [(x[0], x[1], x[2], 365) for x in list(itertools.product(p, d, q))]
s = [ 12]
P = range(0,2)
D = [1]
Q = [1,2]
seasonal_pdq = list(itertools.product(P, D, Q,s))
warnings.filterwarnings("ignore")
l = len(list(itertools.product(pdq,seasonal_pdq)))
pdqs = list(itertools.product(pdq,seasonal_pdq))

# Store results here
ETI_min_aic = {}
for site in b.columns.tolist():
#for site in skippedsites:
    try:
        print('ARIMA parameter search for {}'.format(site))
        # Set data
        data =  b.loc[:,site].squeeze()
        scaler = StandardScaler()
        scaler.fit(data.dropna().values.reshape(-1,1))
        data = data.dropna().apply(lambda x: scaler.transform(x)[0][0])
        data = data.resample('MS').asfreq()
        #data = getLongestRec( data )    
        #data = data.interpolate(method='cubic')
        #Run ARIMA grid search
        AIC_list = pd.DataFrame({}, columns=['param','param_seasonal','AIC'])
        for i,params in enumerate(pdqs):
                try:
                    mod = sm.tsa.statespace.SARIMAX(data,
                                                    order=pdqs[i][0],
                                                    seasonal_order=pdqs[i][1],
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    
                    results = mod.fit()
                    
                    #print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                    temp = pd.DataFrame([[ pdqs[i][0] ,  pdqs[i][1] , results.aic ]], columns=['param','param_seasonal','AIC'])
                    AIC_list = AIC_list.append( temp, ignore_index=True)  
                    del temp           
                except:
                    continue
                
        # Get best AIC score and plot
        m = np.nanmin(AIC_list['AIC'].values) # Find minimum value in AIC
        l = AIC_list['AIC'].tolist().index(m) # Find index number for lowest AIC
        Min_AIC_list = AIC_list.iloc[l,:]
        mod = sm.tsa.statespace.SARIMAX(data,
                                        order= Min_AIC_list['param'],
                                        seasonal_order=Min_AIC_list['param_seasonal'],
                                        enforce_stationarity=False,
                                        enforce_intertibility=True)
        results=mod.fit()
        datapred = results.predict(start = data.first_valid_index() , end= pd.datetime(2017,9,1))
        datapred = scaler.inverse_transform( datapred )
        data = data.apply(lambda x: scaler.inverse_transform([x])[0])
        #results.summary()
        results.plot_diagnostics(figsize=(8,8))
        
        
        # Make Plots
        plt.suptitle('{} {}'.format(site[0],site[1]))
        fig2 = plt.gcf()
        
        fig, ax = plt.subplots(1,1,figsize=(14,3.5))
        ax.plot(data,linestyle=':',linewidth=1.2,c='k')
        #ax.plot(data.interpolate(method='cubic').iloc[data.isnull().nonzero()],
        #        marker='o',linestyle='',markeredgecolor='k',markerfacecolor='darkorange')
        ax.plot(pd.date_range(start=data.first_valid_index(),end=pd.datetime(2017,9,1),freq='MS'),
                    datapred,linewidth=1,alpha=0.8)
        
        ETI_min_aic[ site ] = {'Params': Min_AIC_list,
                               'fig_data': fig,
                               'fig_diag': fig2,
                               'results': results.summary()}
        plt.close('all')
    # Write file if something goes wrong
    except:
        with open('ETIdaho_ARIMA_gridsearch_results.pkl', 'wb') as f:
            pickle.dump(ETI_min_aic, f, protocol=pickle.HIGHEST_PROTOCOL)
            
# Save to Pickle File
with open('ETIdaho_ARIMA_gridsearch_results.pkl', 'wb') as f:
    pickle.dump(ETI_min_aic, f, protocol=pickle.HIGHEST_PROTOCOL)

# Spruce up the data plot
for key in d.keys():
    print(key)
    params = d[key]['Params']
    fig = d[key]['fig_data']
    ax = fig.get_axes()[0]
    ax.set_title('{} {} - {}{} AIC:{:3.2f}'.format(*key,*params),fontsize=11)
    plt.tight_layout()
    d[key]['fig_data'] = fig


# Save figures
for key in d.keys():
    keydir = 'D:/ESRP/RechargeData_Alex/FPT/Wetlands/ETIdahoARIMA/' + str(key)
    if not os.path.isdir(keydir):
        os.makedirs(keydir)
    d[key]['fig_data'].savefig(os.path.join(keydir,'ETresults.png'))


    
for key in d.keys():
    data =  b.loc[:,key]   
    scaler = StandardScaler()
    scaler.fit(data.dropna().values.reshape(-1,1))
    data = data.dropna().apply(lambda x: scaler.transform(x)[0][0])
    data = data.resample('MS').asfreq()
    
    params = d[key]['Params']
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order= params[0],
                                    seasonal_order=params[1],
                                    enforce_stationarity=False,
                                    enforce_intertibility=True)
    results = mod.fit()
    datapred = results.predict(start = data.last_valid_index(), end = pd.datetime(2017,9,1))
    datapred = scaler.inverse_transform( datapred )

# Load ARIMA dictionary
with open('D:/ESRP/RechargeData_Alex/FPT/Wetlands/ETIdaho_ARIMA_gridsearch_results.pkl','rb') as f:
    dARIMA = pickle.load(f)
    
# Write a stressperiod based csv. Use modeled values only when needed
# SP449 = Sep 2017    
    
stress_periods = ['SP'+str(x) for x in  np.arange(417,450)]
stress_months = pd.date_range(end=pd.datetime(2017,9,1),freq='MS',periods = len(stress_periods))

def ForecastTS( data , ARIMAparams ):
    # Scale Data
    scaler = StandardScaler()
    scaler.fit(data.dropna().values.reshape(-1,1))
    datanorm = data.dropna().apply(lambda x: scaler.transform(x)[0][0])
    datanorm = datanorm.resample('MS').asfreq()
    # Run model
    mod = sm.tsa.statespace.SARIMAX(datanorm,
                                    order= ARIMAparams[0],
                                    seasonal_order=ARIMAparams[1],
                                    enforce_stationarity=False,
                                    enforce_intertibility=True)
    results=mod.fit()
    # Make predictions
    datapred = results.predict(start = data.first_valid_index() , end= pd.datetime(2017,9,1))
    datapred = pd.Series( scaler.inverse_transform( datapred ), 
                         index = datapred.index,
                         name='predict')
    # Remove negative ET, replace with min of real data
    datapred = datapred.where(datapred > 0 , data.min())
    
    # Place predicted data where data is NAN
    data = pd.concat([data,datapred],axis=1) 
    data = data.iloc[:,0].where(np.isfinite(data.iloc[:,0] ),data['predict'])
    
    return data
    
    
    
ETout = {}
for key in dARIMA.keys():
    print('Running {} {} ARIMA'.format(key[0],key[1]))
    data = b.loc[:,key[0:2]].squeeze()
    ETout[key] = ForecastTS(data, dARIMA[key[0:2]]['Params'])

df = pd.concat(ETout,axis=1)
df.columns = b.columns
df.rename(columns={'Wetlands--narrow stands':'Narrow_Include','Wetlands--large stands':'Wide_Include'},level=1,inplace=True)  
df.columns.set_names(['StationNam','COMB_CLASS','ETStation'],inplace=True)
a = df['2015-1-1':]
a.index = pd.MultiIndex.from_tuples(tuple(zip(stress_periods,a.index)))
a.T.to_csv(r'Wetlands_ET_mm_monthly_ARIMA.csv',float_format='%3.2f')


### Sites to run that were missed in ARIMA grid search
skippedsites = [('AMERICAN_FALLS_1_SW', 'Wetlands--large stands'),
 ('IDAHO_FALLS_FAA_ARPT', 'Wetlands--large stands'),
 ('PICABO', 'Wetlands--large stands'),
 ('REXBURG_RICKS_COLLEGE', 'Wetlands--large stands'),
 ('AMERICAN_FALLS_1_SW', 'Wetlands--narrow stands'),
 ('IDAHO_FALLS_FAA_ARPT', 'Wetlands--narrow stands')]