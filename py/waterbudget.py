# -*- coding: utf-8 -*-
"""
For plotting ESPAM Water Budget Data

@author: amoody
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec 

plt.style.use('seaborn-darkgrid')
root = r'D:/ESRP/RechargeData_Alex'
if os.path.isdir(root):
    sys.path.append(root)
else:
    root = r'P:/AMoody/ESRP/RechargeData_Alex'
    if os.path.isdir(root):
        print('Assuming you are working remotely, switch path to %s'%root)

# Read DIV workbooks
f = os.path.join(root,'DIV','ESPAM22_DIVS_WORK_FILE_20180216.xlsx')        
with pd.ExcelFile(f) as xls:
    sheets = xls.sheet_names
    with open(os.path.join(root,'DIV','ESPAM2x_201709.csv'),'w') as fout:
        print('Writing...')
        for sheet in sheets:
            if 'IESW' in sheet:
                print('{}'.format(sheet))
                df = xls.parse(sheet, nrows= 42, usecols =12,header= None ) 
                df.iloc[3:,1:] = df.iloc[3:,1:].astype(float)
                #df.to_csv( fout, header=False, index=False , na_rep='0',float_format='%0.2f')
                dfret = df.iloc[1:,:].copy() # Make return dataframe
                dfret.iloc[2:,1:] = 0
                dfret.iloc[0,1] = 'Gross Returns'
                #dfret.to_csv( fout, header=False, index=False, na_rep ='0',float_format='%0.2f')
                # Concatenate divs and returns
                df = pd.concat([df,dfret])
                df.reset_index(drop=True,inplace=True)
                # Incrementing index 
                df[13] = pd.Series(list(range(1,len(df)+1))).values
                # Yearly gross
                df[14]= np.nan
                df.iloc[[0,1,42],14] = ['ENTITY NAME','GROSS DIVERSIONS','GROSS RETURNS']
                df.iloc[3:41,14]=df.iloc[3:41,1:12].sum(axis=1)
                df.iloc[44:,14] = df.iloc[33:,1:12].sum(axis=1)
                #Write file
                df.to_csv( fout, header=False, index=False , na_rep='0',float_format='%0.2f')
xls.close()                
## WATERBUDGET        
### Annual recharge by water source in AF (TRB,PCH,NIR,CNL,SW INC RECH)

### Annual discharge by WY ( URB, OFF-EXCH-MUD,CIR GW, WET, SNAKE ABV MINI, KIM to KING)
        
### P and ET on Irrigated Lands
        
### Canal seepage and Recharge in surface water entities
### GW Irrigation CIR, Offiste pumping, urban pumping, exchange pumping, mudlake pumping

### Non irrigated recharge
        
### Wetlands discharge (FPT)
### Non snake perched river seepage
### TRB       
### CIR and deficit irrgation

## ARIMA vs Average        