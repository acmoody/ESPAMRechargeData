# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:44:17 2018

@author: amoody
"""

import os 
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.append(r'D:/ESRP/RechargeData_Alex/')

WBROOT = r'D:/ESRP/RechargeData_Alex/'
datapath = os.path.join( WBROOT, 
                        'IWRRI/ESPAM2.2 Water Budget Files',
                        'FPT folder/Exchange Well Data/',
                        'WRA_20180226_2e0a5/')
with open(os.path.join( WBROOT, 'config', 'espam22_waterbudget.yaml' )) as f:
    conf = yaml.safe_load(f)

FPT = conf['FPT']

d = {}

for key in FPT.keys():
    ID = FPT[key]['SiteID']
    fname = datapath + str(ID) + '_history.csv'
    df = pd.read_csv(fname,
                parse_dates = True,
                usecols = ['Flow (CFS)', 'HSTDate'],
                index_col = 'HSTDate')
                
    dfMonth = round(df.resample('M').sum().mul(1.9835 * 43560 ))
    #dfMonth[ dfMonth.isnull() ] = 0
    dfMonth[dfMonth > 0] = dfMonth.mask(dfMonth == 0).mul(-1)
    d[ key ] = dfMonth

#%%
f = pd.DataFrame(data=d['E1'])
f.columns = ['E1']

for i,key in enumerate(d.keys()):
    if i > 0:
        d[ key ].columns = [ key ]
        f= pd.merge( f, d[ key ], left_index=True, right_index=True, how='outer')
        

f = f[(f.index > pd.datetime(2015,12,31)) & (f.index < pd.datetime(2017,10,1))  ]
#f = f.replace(np.nan,0)
ftrans = f.T
ftrans.columns = ['S' + str(sp) for sp in range(429,450)]
