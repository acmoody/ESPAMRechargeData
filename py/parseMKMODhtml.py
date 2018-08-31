# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:33:55 2018

@author: amoody
"""
# FUNCTIONS    
from itertools import product
def table_to_2d(table_tag):
    '''
    from 
    https://stackoverflow.com/questions/48393253/how-to-parse-table-with-rowspan-and-colspan
    '''

    rowspans = []  # track pending rowspans
    rows = table_tag.find_all('tr')

    # first scan, see how many columns we need
    colcount = 0
    for r, row in enumerate(rows):
        cells = row.find_all(['td', 'th'], recursive=False)
        # count columns (including spanned).
        # add active rowspans from preceding rows
        # we *ignore* the colspan value on the last cell, to prevent
        # creating 'phantom' columns with no actual cells, only extended
        # colspans. This is achieved by hardcoding the last cell width as 1. 
        # a colspan of 0 means “fill until the end” but can really only apply
        # to the last cell; ignore it elsewhere. 
        colcount = max(
            colcount,
            sum(int(c.get('colspan', 1)) or 1 for c in cells[:-1]) + len(cells[-1:]) + len(rowspans))
        # update rowspan bookkeeping; 0 is a span to the bottom. 
        rowspans += [int(c.get('rowspan', 1)) or len(rows) - r for c in cells]
        rowspans = [s - 1 for s in rowspans if s > 1]

    # it doesn't matter if there are still rowspan numbers 'active'; no extra
    # rows to show in the table means the larger than 1 rowspan numbers in the
    # last table row are ignored.

    # build an empty matrix for all possible cells
    table = [[None] * colcount for row in rows]

    # fill matrix from row data
    rowspans = {}  # track pending rowspans, column number mapping to count
    for row, row_elem in enumerate(rows):
        span_offset = 0  # how many columns are skipped due to row and colspans 
        for col, cell in enumerate(row_elem.find_all(['td', 'th'], recursive=False)):
            # adjust for preceding row and colspans
            col += span_offset
            while rowspans.get(col, 0):
                span_offset += 1
                col += 1

            # fill table data
            rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or len(rows) - row
            colspan = int(cell.get('colspan', 1)) or colcount - col
            # next column is offset by the colspan
            span_offset += colspan - 1
            value = cell.get_text()
            for drow, dcol in product(range(rowspan), range(colspan)):
                try:
                    table[row + drow][col + dcol] = value
                    rowspans[col + dcol] = rowspan
                except IndexError:
                    # rowspan or colspan outside the confines of the table
                    pass

        # update rowspan bookkeeping
        rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}

    return table

# ---
#  2
# ---

#
#with open(file,'r') as f:
#    doc = parse(f)
#
#d = {}
#for i,t in enumerate(doc.xpath('body/table')):
#    rows = t.findall('tr')
#    data = list()
#    for row in rows: 
#        data.append([c.text for c in row.getchildren()])    
#    d['Table {}'.format(i + 1)] = data 
#
#d2 = {}
#for key in d:
#    table = d[key]
#    d2[key] = pd.DataFrame(table)
#    
# ----
# Beautiful Soup
# ----

file  = 'D:\\ESRP\\RechargeData_Alex\\MKMOD_2x\\ESPAM2x_201709.htm' 
from bs4 import BeautifulSoup    
import pandas as pd
with open(file,'r') as f:
    soup = BeautifulSoup(f.read(),'lxml')
    
d={}
# Parse tables to arbitrarily index dataframe
for i,table in enumerate(soup.find_all('table')):
    d['Table {}'.format(i+1)] = pd.DataFrame(table_to_2d(table))
    

d2={}
for key in d.keys():
    if key not in 'Table 1':
        df = d[key]
        droprow = []
        try:
            [droprow.append(row[0]) for row in df.iterrows() if row[1].str.contains(r'\(\d{1,3}\)|Average|State').sum() ]
            df.drop(labels=droprow, axis=0, inplace=True)
            # Get header lines
            header = pd.MultiIndex.from_arrays(df.iloc[:droprow[0]].values)
            # Get location of first timestamp
            irow = (df.values == '05/1980').nonzero()[0][0]
            icol = (df.values == '05/1980').nonzero()[1][0]
            index = df.iloc[irow:,icol].transform(lambda x: pd.datetime.strptime(x,'%m/%Y'))
            
            # Place in new dataframe
            df2 = pd.DataFrame(index=index, columns=header[icol+1:], data=df.iloc[irow:,icol+1:].values)
            df2 = df2.astype(float)
        except:
            print('{} failed to simplify'.format(key))
            df2 = df
            
        d2[key] = df2

import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
plt.style.use('bmh')

df = d2['Table 2']
df2=df.filter(regex='Efficiency|CIR|Irrigation|Soil|Applied')