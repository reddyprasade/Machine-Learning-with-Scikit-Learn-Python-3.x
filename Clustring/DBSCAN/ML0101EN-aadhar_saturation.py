# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import pandas as pd
from itertools import groupby

import matplotlib.cm

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import numpy as np

data = pd.read_csv('aadhar_saturation.csv')
states_group = data.groupby(by = 'State Name')
sat_list = []

for key, group in states_group:
    sat = 0
    for row in group.iterrows():
        sat += row[1][4]
    sat_list.append((key,sat))
print(sat_list)

fig, ax = plt.subplots() 
m = Basemap(resolution='c', projection='merc', lat_0=54.5, lon_0=-4.36, llcrnrlon=68., llcrnrlat=6., urcrnrlon=97., urcrnrlat=37.)

m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
m.drawcoastlines()

m.readshapefile('India_SHP/INDIA','INDIA')

satlist = []

for state_info in m.INDIA_info:
    state = state_info['ST_NAME']
    sat1 = 0
    
    for x in sat_list:
        if x[0] == state:
            sat1 = x[1]
            break
    satlist.append(sat1)

df_poly = pd.DataFrame({'shapes':[Polygon(np.array(shape), True) for shape in m.INDIA],
                       'area':[area['ST_NAME'] for area in m.INDIA_info],
                       'satlist': satlist})
    
shapes = [Polygon(np.array(shape), True) for shape in m.INDIA]
cmap = plt.get_cmap('Oranges_r')

pc = PatchCollection(shapes, zorder=2)

norm = Normalize()
pc.set_facecolor(cmap(norm(df_poly['satlist'].fillna(0).values)))
ax.add_collection(pc)
    
mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
mapper.set_array(satlist)
plt.colorbar(mapper, shrink=0.4)

ax.set_title('Aadhar Saturation')
plt.rcParams['figure.figsize'] = [15,15]
plt.show()