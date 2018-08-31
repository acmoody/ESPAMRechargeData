


import os
import sys 
import arcpy
import glob
import numpy as np 

# Replace a layer/table view name with a path to a dataset (which can be a layer file) or create the layer/table view within the script
# The following inputs are layers or table views: "FPT\WetlandsPolygons", "FPT\p3940_201504_et.img"
METRICpath = r'X:\\Spatial\\Metric'
METRICsubdir = r'p3940_monthly_et_mosaics'


years = [2015,2016,2017]
for year in years:
	for METRICrast in glob.glob(os.path.join(METRICpath,str(year),METRICsubdir,r'p3940*.img')):
		print(f)
		# Get month and year
		datestr = re.search(r'(\d{4})(\d{2})',f).groups()
		print(datestr)
		yearstr = datestr[0]
		monthstr = datestr[1]


		FPTtable = 'WetlandsPolygons'	# Table to add calculated volumes to
		ETtable = 'ETtemp'
		# Sum ET by Wetland Polygon
		arcpy.gp.ZonalStatisticsAsTable_sa("FPT/WetlandsPolygons",
			"FPT_NAME",
			METRICrast,
			"C:/Users/amoody/Documents/ArcGIS/Default.gdb/{}".format(ETtable),
			"DATA",
			"SUM")

		# Resulting table has FPT_NAME, ZONE_CODE, COUNT, AREA (m^2), and SUM(mm of ET) fields
		# Need to multiply A * (ET/1000)* 35.31 to get volume
		# Calculate field?

		# Process: Add Field to FPT table
		arcpy.AddField_management(FPTtable, "D{}{}".format(yearstr,monthstr), "DOUBLE", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
		# Process: Add Join. Join temp table with Wetlands shape
		arcpy.AddJoin_management(FPTtable, "FPT_NAME", ETtable, "FPT_NAME", "KEEP_ALL")
		
		# Process: Calculate Field
		arcpy.CalculateField_management(FPTtable, "{}.D{}{}".format(ETtable,yearstr,monthstr), "35.31 * !{}.AREA! * !{}.SUM! /1000".format(ETtable), "PYTHON_9.3", "")
		# Process: Remove Join
		arcpy.RemoveJoin_management(FPTtable, "")


		# Replace a layer/table view name with a path to a dataset (which can be a layer file) or create the layer/table view within the script