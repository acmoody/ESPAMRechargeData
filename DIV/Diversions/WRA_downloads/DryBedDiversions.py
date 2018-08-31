# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:18:03 2018

@author: amoody
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
#%%
d = {}
for f in glob.glob('*_history.csv'):
    df = pd.read_csv(f,header=0,parse_dates=True,
                             infer_datetime_format=True,
                             index_col = ['HSTDate'],
                             usecols=['HSTDate', 'SiteType','Flow (CFS)'])
    sitetype = df['SiteType'][0] 
    siteid = str(f[0:8])
    df.drop(labels=['SiteType'],axis=1,inplace=True)
    df.columns = pd.MultiIndex.from_tuples([(siteid,sitetype)],names=('SiteID','Type'))
    d[siteid] = df
    
del df

for i,key in enumerate(d.keys()):
    if i == 0:
        df = d[key]
    else:
        df = df.merge(d[key],right_index=True,left_index=True,how='outer')
        
div_idx = df.columns.get_level_values(1).str.contains('D')
dfdiv = df.filter(regex='D',axis=1)
div = dfdiv['2014-1-1':].resample('M').mean().sum(axis=1)
div = dfdiv['20150101':].resample('M').mean().sum(axis=1)
dfall = df.sum(axis=1).resample('M').sum()
