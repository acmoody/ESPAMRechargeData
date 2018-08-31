# -*- coding: utf-8 -*-
"""
Created on Wed May 23 08:57:49 2018

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
        
# Plot defition
def plotDIV( s ):
    """ Standard presentation plot for ESPAM diversion"""
    fig = plt.figure(tight_layout=True, figsize=(9,9))
    gs = gridspec.GridSpec(2,2)  
      
    # Time Series 
    ax1 = fig.add_subplot(gs[0,0:]) 
    s['2000-1-1':].plot(c='k',linewidth=1,ax = ax1)
    # Monthly  +/- 2 sigma
    ax2 = fig.add_subplot(gs[1,0])
    ax2.fill_between(s.groupby(s.index.month).min().index,
           s.groupby(s.index.month).mean() - 2 * s.groupby(s.index.month).std(),
           s.groupby(s.index.month).mean() + 2 * s.groupby(s.index.month).std(),
           facecolor=[0.7, 0.7,0.7],
           alpha=0.7)
    for yr in np.unique(s['2015-1-1':].index.year):
        idx = s.index.year == yr
        ax2.plot(s[idx].index.month,s[idx].values,
                 label=str(yr),
                 marker='o')
        
    ax2.legend()
    return 
#%%
# DIV + PCH
## IESW000 Misc
### Waiting on 
## IESW005 Big Lost
f = os.path.join(root,'PCH','Big Lost River','Daily_Bryce_DataGaps_Filled with IESW005 csv out 20180522.xlsx')
df = pd.read_excel(f,
                   sheetname = 'Daily_Bryce_DataGaps_Filled.csv',
                   header = 0,
                   parse_dates={'Date':[0,1]},
                   date_parser = lambda x,y:pd.datetime(int(x),int(y),1),
                   index_col='Date')
fig1,ax = plt.subplots()
ax.plot(df['IESW005_1000af'])
#%%
def wateryear( df ): 
    ''' Add water year column to dataframe with datetime index'''
   return [i.year+1 if i.month > 9 else i.year for i,row in df.iterrows()]
   
def wateryearmonth(df):
    return [i.month - 9 if i.month > 9 else i.month + 3 for i,row in df.iterrows()]


#IESW0005
df['wateryear'] = wateryear(df)
df['wymonth'] = wateryearmonth(df)

fig, ax = plt.subplots(1,figsize=(6,6))
df['IESW005'].groupby(df.wateryear,df.index.month).mean().plot(ax=ax,linewidth=1,c=[0.5, 0.5,0.5])

ax.fill_between(df['IESW005'].groupby(df.index.month).min().index,
       df['IESW005'].groupby(df.index.month).min().values,
       df['IESW005'].groupby(df.index.month).max().values,
       facecolor='grey')


from sklearn.linear_model import LinearRegression, TheilSenRegressor
df.drop(labels=['SILVER CREEK AT SPORTSMAN ACCESS NR PICABO ID'],inplace=True,axis=1)

boo = df.notnull().all(axis=1)

f = lambda x: (x -x.mean())/x.std()

dfshift = df - df.shift(1)
dfshift.corr()
df[boo]
dfm = df.resample('M').mean()
plt.scatter( dfm['LITTLE WOOD RIVER NR CAREY ID'], dfm['FISH CREEK NR CAREY ID'].shift(-1))

df[df == 0 ] = 0.001
fc = boxcox(df[boo]['FISH CREEK NR CAREY ID'])[0]
lw = boxcox(df[boo]['LITTLE WOOD RIVER NR CAREY ID'])[0]

#%%
q_pch = pch[7]['1981-1-1':]
q_div = div['cfs']['1981-1-1':]


plt.style.use('bmh')
fig,ax=plt.subplots(2,1,sharex=True)
ax[0].plot(q_pch,label='PCH')
ax[0].plot(q_div,label='DIV')
ax[0].legend()
ax[0].set_ylabel('cfs')

ax[1].plot((q_div.cumsum()-q_pch.cumsum()).cumsum() *59.5/ 100)
ax[1].set_ylabel('100s AF')
ax[1].set_title('Cumulative DIV-PCH',fontsize=11)
#%%
ax[1].plot(q_div.cumsum())
ax[1].plot(q_pch.cumsum())

#%% Read in monthly NWIS stats from clipboard
df = pd.read_clipboard(header=None,parse_dates={'Date':[4,5]},date_parser=lambda x,y:pd.datetime(int(x),int(y),1),index_col='Date')
df.drop(labels=[0,1,2,3],axis=1,inplace=True)
