import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyabc
from pyabc.visualization import plot_kde_2d, plot_data_callback, plot_kde_1d_highlevel
import os
import tempfile
from model import modelCov

dcOb = np.array([2249, 2292, 3098, 3039, 2544, 2443, 3859,
3302, 3331, 3593, 3289, 2666, 2141, 3392,
3548, 4360, 4629, 4959, 4854, 5320, 5604,
6403, 7059, 7603, 7241, 6756, 6971, 9927,
10235, 12555, 13188, 13692, 11544, 11805,
16547, 17061, 18331, 18242, 15729, 12516,
12070, 19447, 18892, 19708, 18438, 17724,
14762, 14191, 25616, 25329, 25447, 23250,
21478, 16242, 15725, 26546, 24098, 23611,
23352, 22678, 16490
])

dOb = np.array([41649, 41658, 41668, 41675, 41688, 41696, 41711,
41720, 41729, 41742, 41755, 41773, 41789, 41810,
41828, 41854, 41881, 41905, 41925, 41957,
41984, 42023, 42079, 42114, 42146, 42185, 42232,
42286, 42334,42391,42458,
42526,42593,42653,42725,42796,42898,42990,43069,
43172,43295,43408,43518,43634,43785,43935,44112,
44271,44462,44657,44878,45114,45336,45550,45799,
46077,46344,46630,46941,47279,47599
])

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test1.db"))

h = pyabc.History(db_path)

df, w = h.get_distribution()

pars={}
for x in range(0,15):
    pars["{}".format(df.columns[x])] = np.average(df[df.columns[x]])


days = len(dcOb)
measurement_times = np.arange(days)
daysd = np.linspace(0, days, days)

N, S, E, IA, IAD, IS, ISD, D, R, I, IDD, G = modelCov(pars,days)
"""
IDD.pop(0)
D.pop(0)
I.pop(0)"""

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(daysd, dcOb, 'y', alpha=0.5, lw=2, label='Observed Daily Cases')
ax.plot(daysd, IDD, 'b', alpha=0.5, lw=2, label='Simulated Daily Cases')
ax.plot(daysd, dOb, 'c', alpha=0.5, lw=2, label='Observed Cumulative Deaths')
ax.plot(daysd, D, 'r', alpha=0.5, lw=2, label='Simulated Cumulative Deaths')
ax.set_xlabel('Number of days since 1st September 2020')
ax.set_ylabel('Number of People')
ax.set_ylim(0,50000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
