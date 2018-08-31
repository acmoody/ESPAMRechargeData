#!/usr/bin/env python
''' Calculate ESPAM wetland recharge/discharge (FPT file for MODFLOW) using PRISM and METRIC ET'''
import os
import glob
import re
#import pandas as pd
#import numpy as np
import datetime as dt
#import tempfile
from grass.script import parser, run_command, parse_command, fatal
from grass.exceptions import CalledModuleError

METRICpath = r'X:\Spatial\Metric'
METRICsubdir = r'p3940_monthly_et_mosaics'

def main():
	years = [2015,2016,2017]
	for year in years:
		for METRICrast in glob.glob(os.path.join(METRICpath,str(year),METRICsubdir,r'p3940*.img')):
			# Get month and year
			datestr = re.search(r'(\d{4})(\d{2})',METRICrast).groups()
			yearstr = datestr[0]
			monthstr = datestr[1]
			print(' Reading METRIC for {}'.format( dt.date(int(yearstr),int(monthstr),1) ) )
			
			tempshp = 'ETvolume_temp'

			# Import ET from METRIC if not imported already
			outname = 'metric_p3940_{}{}'.format(yearstr,monthstr)
			checkrast = parse_command('g.list', type= 'raster', mapset='ESRP', pattern = outname )
			if not checkrast:
				run_command('r.in.gdal',
							overwrite = True,
							input= METRICrast,
							output=outname,
							title='METRIC ET (mm) {}{}'.format(yearstr,monthstr) )

			# Wetland flux = PRISM - ET. Calculate coarser FPT
			PRISM = 'PRISM_{}_{}@PRISM'.format(yearstr,monthstr)
			run_command('g.region',res=30)
			run_command('r.mapcalc', expression = 'PlessET = ({} / 100. ) - {}'.format(PRISM , outname) )
			# Get volume of ET by clumping with wetlands raster( converted from polygons)
			# Volumes will be in m^2 * mm . Upsample to allow small areas less than 900 m^2 get included
			# in volume calcs.
			run_command('g.region',res=5)
			run_command('r.volume',
						overwrite = True,
						quiet= True,
						input= 'PlessET',
						clump='Wetlands@ESRP',
						centroids= tempshp )
			# Delete PlessET
			run_command('g.remove',type='raster',pattern='PlessET',flags='f')
			# Tempshp has fields of volume,average,sum,and count
			# Join column
			run_command('v.db.join', map='Wetlands_bound@ESRP', column='CELL_INTGR', 
				other_table=tempshp, other_column='cat', subset_columns='volume')

			# Make new column with timestamp info (ET201510 or whatever)
			run_command('v.db.addcolumn', map='Wetlands_bound@ESRP', columns='FPT_{}{} double precision'.format(yearstr,monthstr) )
			# Update with volume col conerted to cubic feet (35.31 cubic feet per cubic meter)
			#run_command('v.db.update' , map ='Wetlands@ESRP', layer=1,
			#	column='ET_{}{} double precision'.format(yearstr,monthstr), 
			#	query_column= 'volume * 35.31 / 1000' )
			run_command('db.execute', sql=' UPDATE Wetlands_bound SET FPT_{}{} =  volume  * 35.31 / 1000'.format(yearstr,monthstr) )
			# drop joined column
			run_command('v.db.dropcolumn', map='Wetlands_bound@ESRP', columns='volume')

	# Write output file. I'm assuming the group sums up by cell_intgr
	run_command('v.db.select',
			    overwrite = True,
			    map='Wetlands_bound@ESRP' ,
				columns='FPT_NAME,CELL_INTGR,FPT_201504,FPT_201505,FPT_201506,FPT_201507,FPT_201508,FPT_201509,FPT_201510,FPT_201604,FPT_201605,FPT_201606,FPT_201607,FPT_201608,FPT_201609,FPT_201610,FPT_201704,FPT_201705,FPT_201706,FPT_201707,FPT_201708,FPT_201709,FPT_201710 ',
				group='CELL_INTGR',
				separator='comma',
	 			file=r'D://ESRP//RechargeData_Alex//FPT//Wetlands//wetlands_out.csv' )
if __name__ == '__main__':
	main()