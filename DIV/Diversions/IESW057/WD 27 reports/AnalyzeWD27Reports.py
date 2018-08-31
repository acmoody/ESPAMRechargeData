# -*- coding: utf-8 -*-
"""
Scipt to analyze IESW 057 data from data sent over by Water District 27. The
data entry sheet in YYYY Blackfoot River Weekly Accounting excel sheets contains
daily diversions for all of the diversions in the district. Here we aggregate 
them into water year totals to compare to years prior to WY 2014, when we were
not provided the data.

IESW057 is comprised of the Fort Hall Project diversions, WD27 Middle Region 
users, Smith (13068499), and the Blackfoot diversions (13061430)


@author: amoody
"""
import numpy as np
import pandas as pd
import glob

fdir = r'D:/ESRP/RechargeData_Alex/DIV/Diversions/IESW057/WD 27 reports/*Weekly Accounting*'

d = {}
for f in glob.glob( fdir ):
    fyear = int(re.search('\d{4}',f.split('\\')[-1]).group() )
    print(f)
    df = pd.read_excel( f, sheet_name='Data Entry',      
                       error_bad_lines=False,
                       warn_bad_lines=True,
                       parse_dates=True )
    
    # Get header row. This seems to be the most consistent string for the header
    try:
        hrow = df.index.get_locs([('Region', 'Diversion Name')])[0]
    except:
        hrow = [int(r[0]) for r in df.iterrows() if r[1].str.contains('Diversion Name').sum()]
        hrow=hrow.pop()
        new_idx =  pd.MultiIndex.from_tuples(tuple(df.iloc[6: ,0:5].values))
        df = df.iloc[hrow:,5:]
        df.index = new_idx
        hrow=0
        
    
    # Get index of last valid column 
    hcol = (df.columns.str.contains('Unnamed') == True).nonzero()[0][0] 
    
    df2 = df.iloc[hrow:,:hcol].T
    df2.index =  pd.date_range(
            start = pd.datetime(fyear,4,1),
            periods=len(df2),
            freq='D')
    
    divs = ['Middle Region Total',
            '13068499',
            'Little Indian',
            'Main Canal',
            'North Canal']
    
    cols = []        
    colnames = []
    for level in range(0,5):
        for s in divs:
            strcheck = df2.columns.get_level_values(level).astype(str).str.contains(s)
            strsum = strcheck.sum()
            if strsum:
               colnames.append(s)
               cols.append(strcheck.nonzero()[0][0])
    
    df2 = df2.iloc[:,cols]
    df2.columns = colnames
    d[fyear]=df2
              
del df
df = pd.concat(d.values(),axis=0)
df = df.sort_index()
df.replace({np.nan:0},inplace=True)
# Read in Blackfoot Canal Data ( WRA 13061430 )
f= r'D:/ESRP/RechargeData_Alex/DIV/Diversions/IESW057/WD 27 reports/13061430_Blackfoot_WRA_diversions/13061430_history.csv'
df3 = pd.read_csv(f, usecols=[2,4],index_col=1)

df = pd.merge(df,df3,how='inner',right_index=True,left_index=True)

dfKAFM = df.resample('M').sum().apply(lambda x: x*1.9835/1000).sum(axis=1)
#data =