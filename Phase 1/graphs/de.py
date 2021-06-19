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
                   41647
                   ])

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test2.db"))

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
ax.set_title('Observed vs Simulated Cumulative Deaths 1/8/20 - 31/8/20')
ax.set_xlabel('Number of days since 1st August 2020')
ax.set_ylabel('Number of People')
ax.set_ylim(41200,42000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
