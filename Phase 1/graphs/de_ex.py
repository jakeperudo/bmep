import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyabc
from pyabc.visualization import plot_kde_2d, plot_data_callback, plot_kde_1d_highlevel
import os
import tempfile
from model import modelCov

dOb = np.array([41369, 41379,
                   41394, 41408, 41414, 41423, 41435, 41449, 41457,
                   41470, 41480, 41487, 41496, 41503, 41515, 41519,
                   41531, 41540, 41543, 41551, 41558, 41566, 41580,
                   41586, 41598, 41609, 41617, 41627, 41632, 41638,
                   41647,
                   41649, 41658, 41668, 41675, 41688, 41696, 41711,
                   41720, 41729, 41742, 41755, 41773, 41789, 41810,
                   41828, 41854, 41881, 41905, 41905, 41925, 41957,
                   41984, 42023, 42079, 42114, 42146, 42185, 42232,
                   42286, 42334,42391,42458,
                   42526,42593,42653,42725,42796,42898,42990,43069,
                   43172,43295,43408,43518,43634,43785,43935,44112,
                   44271,44462,44657,44878,45114,45336,45550,45799,
                   46077,46344,46630,46941,47279,47599])

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test4.db"))

h = pyabc.History(db_path)
df, w = h.get_distribution()
p={}
for x in range(0,15):
    p["{}".format(df.columns[x])] = np.average(df[df.columns[x]])

days = len(dOb)

N, S, E, IA, IAD, IS, ISD, D, R, I, IDD, G = modelCov(p,days)

dayssi = np.linspace(0, days, days)
daysob = np.linspace(0, len(dOb), len(dOb))

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(daysob, dOb, 'c', alpha=0.5, lw=2, label='Observed Cumulative Deaths')
ax.plot(dayssi, D, 'r', alpha=0.5, lw=2, label='Simulated Cumulative Deaths')
ax.set_title('Observed vs Simulated Cumulative Deaths 1/8/20 - 31/10/20')
ax.set_xlabel('Number of days since 1st August 2020')
ax.set_ylabel('Number of People')
ax.set_ylim(41200,48000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
