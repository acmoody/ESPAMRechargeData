# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:28:28 2018

@author: amoody
"""

#!/usr/bin/env python
''' Calculate ESPAM wetland recharge/discharge (FPT file for MODFLOW) using PRISM and METRIC ET'''
import os
import sys
import glob
import re
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.warp import Resampling
from rasterstats import zonal_stats
import warnings

# Parallel processors
import multiprocessing
# Add some raster handling tools
sys.path.append(r'D:\TreasureValley\notebooks\py_modules')
from gridtools import resample
# --------------
# SWITCHES 
# --------------
calcFPT=True
loadFPT=False
# -------------
# METRIC Months (Growing Season)
# ---------------------------------
# METRIC Rasters
METRICpath = r'X:\Spatial\Metric'
METRICsubdir = r'p3940_monthly_et_mosaics'
METRICsuff =  r'p3940_{}_et.img'
#PRISM Rasters
PRISMpath = r'X:\Spatial\Precipitation\Monthly'
PRISMsuffix = 'pcp{}_{}_id.img'
# FPT Wetlands shapefile
wetlands_src = r'D:\ESRP\RechargeData_Alex\FPT\Wetlands\Shapefiles\WetlandsPolygons\WetlandsPolygons.shp'
dfWetlands = gpd.GeoDataFrame.from_file(wetlands_src)

# ---------------------
# ETidaho ARIMA models
# ----------------------
'''
A forecast of ETIdaho for 2017 was made in AnalyzeETIdahoESRP.py using
ARIMA models selected from a gridsearch of parameters. Load this data and
for non-growing season

'''

f = r'Wetlands_ET_mm_monthly_ARIMA.csv'
dfETI = pd.read_csv(f,header=[1],parse_dates =True)

# --------------
# Bookkeeping
# --------------
        
# Add point ID field to attributes
# dfWetlands['POINT_ID'] = dfWetlands.apply((lambda x: x['COMB_CLASS'][0] + str(x['CELL_INTGR'])),axis=1)
# Load shapefile linking ET stations to PointID AKA FPT_Name        
df = gpd.GeoDataFrame.from_file(r'GIS\WetlandsFPT_ETStations.shp')
df.replace({'wide_wet_2011060':'Wide_Include','Narrow_wet_20110':'Narrow_Include'},inplace=True)
#df['ETStation']= df['ETStation'].astype(int)
df=df.iloc[:,~df.columns.str.contains('p_.*')]
df2=df.merge(dfETI,how='left',
             left_on=['ETStaNum','SOURCETHM'],
             right_on=['ETStation','COMB_CLASS'])
df2.drop(labels=['ETStation_x','NAME','SOURCETHM','FLAG','geometry','StationNam_y','COMB_CLASS'],axis=1,inplace=True)
df2.set_index('POINT_ID',inplace=True)
# Now join to attribute table of individual polygons not grouped by cell
dfWetlands = dfWetlands.merge(df2, how='outer',
                          left_on='FPT_NAME',
                          right_index=True)

#----- Backcast with Metric
rasts = glob.glob(r'X:\Spatial\METRIC\*\p3940_monthly_et_mosaics\*.img',recursive=True)
ts = [re.findall(r'.*_(\d{4}\d{2}).*', s)[0] for s in rasts]
ts = [pd.datetime.strptime(s,'%Y%m') for s in ts]
#----------
# CALC FPT
# ----------
# Set up pool
pool = multiprocessing.Pool(4)
if calcFPT:
    warnings.filterwarnings('ignore')
    #Date range for calculations
    #ts = pd.date_range(start='2015-01-01',freq='MS',periods=33)
    for date in ts:
        datestr = date.strftime('%Y%m')
        # PRISM file name and full path
        PRISMf = PRISMsuffix.format(datestr[0:4],datestr[-2:])
        PRISMrast = os.path.join( PRISMpath, str(date.year),PRISMf)
        # ETIDAHO if non-growing season 
        if (date.month < 4) | (date.month > 10):
            
            print('Retriveing ETIdaho for {}'.format(date.date()))
            print('    ... Reading {}'.format(PRISMf))
            print('    ... Getting zonal statistics for {}'.format(PRISMrast))
            stats_prism = zonal_stats(dfWetlands, 
                                      PRISMrast,
                                      stats = 'mean',
                                      all_touched=True)
            dfPr = pd.DataFrame(stats_prism).div(100).squeeze()
            dfET = dfWetlands[datestr]
            dfFPT = (dfPr - dfET) * 3.28e-3 * dfWetlands['ACRES'] * 43560
            dfFPT = pd.DataFrame(dfFPT)
            FPTcol='FPT_ft3_{}'.format(datestr)        
            dfFPT.rename(columns={0:FPTcol }, inplace = True)
            dfWetlands[FPTcol] = dfFPT
        # METRIC if growing season
        else:    
            rast = os.path.join(METRICpath, str(date.year),METRICsubdir,METRICsuff.format(datestr))
            # Get date string of METRIC raster and use to find corresponding PRISM
            #datestr = re.findall('.*(\d{6}).*',rast.split('\\')[-1])[0]
            print('Processing METRIC for {}'.format(datestr))
            print('    ... Reading/resampling {}'.format(PRISMsuffix.format(datestr[0:4],datestr[-2:]) ) )
                
            # PRISM = resample(PRISMrast, 30, method=Resampling.nearest)
            # Resample PRISM. Rasterstats has an 'alltouched' option. This may double count
            # a lot of cells along cell faces
            print('    ... Getting zonal statistics for {}'.format(PRISMrast))
            stats_prism = zonal_stats(dfWetlands, 
                                      PRISMrast,
                                      stats = 'mean',
                                      all_touched=True)
            
            print('   ... Getting zonal statistics for {}'.format(rast))
            stats = zonal_stats( dfWetlands, 
                                rast, 
                                stats = 'sum' ,
                                all_touched=True)
               #del PRISM
               
            #Cell Area * mm in m * ft3 in m3   
            dfET = pd.DataFrame(stats).mul(900 * 35.31 / 1000).squeeze()
            # 100s of mm to mm
            dfPr = pd.DataFrame(stats_prism).div(100).squeeze()
            # mean of P in polygon   * mm to ft  * ft * ft to acre div acre per feet
            dfPr = dfPr.mul(3.28e-3).mul(dfWetlands['ACRES'],axis=0).mul(43560).squeeze()
            dfFPT = (dfPr - dfET)
            dfFPT = pd.DataFrame(dfFPT)
            FPTcol = 'FPT_ft3_{}'.format(datestr)
            dfFPT.rename(columns={0:FPTcol }, inplace = True)
            dfWetlands[FPTcol] = dfFPT
    
    # Drop areas outside of model boundary and sum by FPT 'entity'
    dfFPT = dfWetlands.groupby('FPT_NAME').sum()
    dfFPT = dfFPT.loc[df2.index,:]
    dfFPTp=dfFPT.filter(regex='FPT_.*').T
    dfFPTp.T.to_csv('FPT_ETIDAHO_METRIC_irr_198604_201709_noresamp.csv')
    dfFPTp.index=ts
    
#--------------
# Analysis
#--------------
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.style.use('bmh')

if loadFPT:
    ts = pd.date_range(start='2015-01-01',freq='MS',periods=33)
    f = r'FPT_ETIDAHO_METRIC_201501_201709_noresamp.csv'
    dfFPTn = pd.read_csv(f, header=0, index_col=0) 
    dfFPTn.columns=ts
    # Read in Jennifers FPT work and plot
    f = r'WetlandsETPCP.csv'
    dfFPTo = pd.read_csv(f, header=0,index_col=1)
    dfFPTo = dfFPTo.filter(regex='FPT_.*')
    dfFPTo.columns = pd.date_range(end='2014-12-1',freq='MS',periods=74)
    
FPT = pd.concat([dfFPTo.T,dfFPTn.T],axis=0)
FPTnorm =  FPT.apply(lambda x: (x - x.mean())/x.std())
# Narrow wetlands seem to be less using METRIC lets compar
fig, ax = plt.subplots(1,1,tight_layout=True)
fillidx = (FPTnorm.index.month > 10) | (FPTnorm.index.month < 4)
ax.fill_between(FPTnorm.index,-3,2,where = fillidx,facecolor=[0.8, 0.8, 0.75],alpha=0.5)
ax.plot(FPTnorm.filter(regex='N.*').mean(axis=1))
ax.plot(FPTnorm.filter(regex='W.*').mean(axis=1))
ax.set_ylim(-2.5,1.2)
ax.set_ylabel(r'Mean FPT$_{norm}$')
plt.legend(['Narrow','Wide'])

# Compare 1979-2014 FPT file that uses ETIdaho with METRIC 
fMETRIC = r'WetlandsFPT_METRIC_1986_2017.csv'
dfMETRIC = pd.read_csv(fMETRIC, header=0, index_col=0) 
dfMETRIC = dfMETRIC.filter(regex='FPT_.*')
dfMETRIC = dfMETRIC.groupby('FPT_NAME').sum()
datestr = dfMETRIC.columns.tolist()
datedt = [ pd.datetime.strptime(s[-6:],'%Y%m') for s in datestr]
dfMETRIC=dfMETRIC.rename(columns =  dict(zip(dfMETRIC.columns,datedt)))
dfMETRIC = dfMETRIC.T.sort_index()

stress_periods = ['S{0:03d}'.format(x) for x in  np.arange(1,450)]
#----------------------
# Effects of resampling
# ---------------------
#
## 4 km
#
#stats4km_m = zonal_stats(dfWetlands,PRISMrast,stats='mean',all_touched=True)
#stats4km_s = zonal_stats(dfWetlands,PRISMrast,stats='sum',all_touched=True)
## 30 m
#PRISM = resample(PRISMrast, 30, method=Resampling.nearest)
#stats30m_m = zonal_stats(dfWetlands, PRISM['raster'], 
#                          affine=PRISM['affine'], 
#                          stats = 'count mean',all_touched=True)
#stats30m_s= zonal_stats(dfWetlands, PRISM['raster'], 
#                          affine=PRISM['affine'], 
#                          stats = 'sum',all_touched=True)
#statslist = [stats4km_m, stats4km_s, stats30m_m, stats30m_s]
#x=[pd.DataFrame(s) for s in statslist ]
#statsdf = pd.concat(x,axis=1,keys = ['4km_mean','4km_sum','30m_mean','30m_sum'])
#statsdf.columns = statsdf.columns.droplevel(1)
#statsdf = statsdf.div(100)
## Make copy of dfWetlands
#dfWL = dfWetlands.iloc[:,:14]
#dfWL = dfWL.merge(statsdf, left_index=True, right_index=True)
# Multiply means by area of polygon, multiply sums by pixel resolution


#----- 
# Write to FPT file
# -------
f=r'D:\ESRP\RechargeData_Alex\FPT\ESPAM2_FPT_20180305.csv'
dfMOD = pd.read_csv(f, header= 0, index_col=0)
stress_periods = ['S'+str(x) for x in  np.arange(417,450)]
dfFPTp = dfFPTp.T
dfFPTp.columns = stress_periods
#
temp = dfMOD.loc[dfMOD.index.str.contains('W|N'),:'S416']
temp = temp.merge(dfFPTp,right_index=True,left_index=True)
dfMOD.loc[dfMOD.index.str.contains('W|N'),:] = temp
dfMOD.apply(np.round).to_csv(r'D:\ESRP\RechargeData_Alex\FPT\ESPAM2_FPT_197905_201709.FPT')
FPTtype = dfMOD.index.str[0]
FPTts = dfMOD.apply(lambda x: (x - x.mean())/x.std(),axis=1).groupby(FPTtype).mean().T
FPTts.index = pd.date_range(end='2017-9-1',freq='MS',periods=len(FPTts))
figFPT, ax2 = plt.subplots(1,1)
FPTts['2000-1-1':].plot(ax=ax2,cmap='Accent',subplots=True,layout=(2,3))
ax2.set_ylabel('Mean FPT$_{norm}$')
#with fiona.open(wetlands_src ,'r') as src:
#    records = [r for r in src]
#    geoms = [r['geometry'] for r in records ]
#    attr = [r['properties'] for r in records ]
#    #geoms = [(g, attr[i]['cat']) for i, g in enumerate(geoms) if g]    
#    meta = source.meta
#    sink_schema = source.schema.copy()

#with fiona.open(wetlands_src, 'r')
#def main():
#	years = [2015,2016,2017]
#	for year in years:
#		for METRICrast in glob.glob(os.path.join(METRICpath,str(year),METRICsubpddir,r'p3940*.img')):
#			# Get month and year
#			datestr = re.search(r'(\d{4})(\d{2})',METRICrast).groups()
#			yearstr = datestr[0]
#			monthstr = datestr[1]
#			print(' Reading METRIC for {}'.format( dt.date(int(yearstr),int(monthstr),1) ) )
#			
#			tempshp = 'ETvolume_temp'
#
#			# Import ET from METRIC if not imported already
#			outname = 'metric_p3940_{}{}'.format(yearstr,monthstr)
#			checkrast = parse_command('g.list', type= 'raster', mapset='ESRP', pattern = outname )
#			if not checkrast:
#				run_command('r.in.gdal',
#							overwrite = True,
#							input= METRICrast,
#							output=outname,
#							title='METRIC ET (mm) {}{}'.format(yearstr,monthstr) )
#
#			# Wetland flux = PRISM - ET. Calculate coarser FPT
#			PRISM = 'PRISM_{}_{}@PRISM'.format(yearstr,monthstr)
#			run_command('g.region',res=30)
#			run_command('r.mapcalc', expression = 'PlessET = ({} / 100. ) - {}'.format(PRISM , outname) )
#			# Get volume of ET by clumping with wetlands raster( converted from polygons)
#			# Volumes will be in m^2 * mm . Upsample to allow small areas less than 900 m^2 get included
#			# in volume calcs.
#			run_command('g.region',res=5)
#			run_command('r.volume',
#						overwrite = True,
#						quiet= True,
#						input= 'PlessET',
#						clump='Wetlands@ESRP',
#						centroids= tempshp )
#			# Delete PlessET
#			run_command('g.remove',type='raster',pattern='PlessET',flags='f')
#			# Tempshp has fields of volume,average,sum,and count
#			# Join column
#			run_command('v.db.join', map='Wetlands_bound@ESRP', column='CELL_INTGR', 
#				other_table=tempshp, other_column='cat', subset_columns='volume')
#
#			# Make new column with timestamp info (ET201510 or whatever)
#			run_command('v.db.addcolumn', map='Wetlands_bound@ESRP', columns='FPT_{}{} double precision'.format(yearstr,monthstr) )
#			# Update with volume col conerted to cubic feet (35.31 cubic feet per cubic meter)
#			#run_command('v.db.update' , map ='Wetlands@ESRP', layer=1,
#			#	column='ET_{}{} double precision'.format(yearstr,monthstr), 
#			#	query_column= 'volume * 35.31 / 1000' )
#			run_command('db.execute', sql=' UPDATE Wetlands_bound SET FPT_{}{} =  volume  * 35.31 / 1000'.format(yearstr,monthstr) )
#			# drop joined column
#			run_command('v.db.dropcolumn', map='Wetlands_bound@ESRP', columns='volume')
#
#	# Write output file. I'm assuming the group sums up by cell_intgr
#	run_command('v.db.select',
#			    overwrite = True,
#			    map='Wetlands_bound@ESRP' ,
#				columns='FPT_NAME,CELL_INTGR,FPT_201504,FPT_201505,FPT_201506,FPT_201507,FPT_201508,FPT_201509,FPT_201510,FPT_201604,FPT_201605,FPT_201606,FPT_201607,FPT_201608,FPT_201609,FPT_201610,FPT_201704,FPT_201705,FPT_201706,FPT_201707,FPT_201708,FPT_201709,FPT_201710 ',
#				group='CELL_INTGR',
#				separator='comma',
#	 			file=r'D://ESRP//RechargeData_Alex//FPT//Wetlands//wetlands_out.csv' )
#if __name__ == '__main__':
#	main()