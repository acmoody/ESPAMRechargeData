# -*- coding: utf-8 -*-
"""
WARNING: NOT MEANT TO BE RUN AS A SCRIPT!!!!!

There are just python snippets to duplicate previous stress period in the ETI file. 
We are waiting to update and calculate cell by cell ET, so 2014 was just carried
forward.

Created on Wed Aug 29 13:27:38 2018

@author: amoody
"""
import csv
import re
import pandas as pd
import numpy as np

# Open existing ETI file. This particular one is found in the RechargeData_22
# directory, made by J. Sukow

f = './ALL_ET_10192017.ETI'
with open(f) as handle:
    reader = csv.reader(handle)
    # Empty dictionary to hold data
    d = {}
    # Skip headers, and lots of the data for that matter
    for i in range(0,42828):
        next(reader)
    # loop rows.
    for row in reader:       
        # If this is the start of a stress period, save that period
        if 'STRESS PERIOD' in row[0]:
            sp = row[0]
            print(sp)
            results = []
        # Append data
        elif len(row) > 2:
            results.append( np.array(row).astype(float))
        
        # Convert 2d array and place in data dictionary
        df = pd.DataFrame(results)
        print('    Writing dataframe for {}'.format(sp))
        d[ sp ] = df
                
            
a=['STRESS PERIOD {}'.format(s) for s in range(417,417+12)]

# Copy previous years forward
# Empty dictionary to hold copied data
d2 = {}
# Adding 3 years, so use a multiplier
for i in range(1,4):
    for key in d.keys():
        sp = re.findall(r'\d{3}',key)[0]
        # New SP is the same month in the next year
        newsp = int(sp) + 12 * i
        # Stopping in 2017-09 (SP44())
        if newsp < 450:
            d2['STRESS PERIOD {}'.format(newsp)]=d[key]
            
# Write to a file. This can then be pasted into the existing ETI file.
with open('ETI_append_417-449.csv','w') as fout:
    for key in d2.keys():
        meta = pd.Series([key,1])
        meta.to_csv(fout, index=None,header=None)
        d2[key].to_csv(fout, index=None,header=None)
