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
                   3548, 4360, 4629, 4959, 4854, 5320, 5604
                   ])

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test1.db"))

h = pyabc.History(db_path)
df, w = h.get_distribution()
p={}
for x in range(0,15):
    p["{}".format(df.columns[x])] = np.average(df[df.columns[x]])

days = len(dcOb)

N, S, E, IA, IAD, IS, ISD, D, R, I, IDD, G = modelCov(p,days)

dayssi = np.linspace(0, days, days)
daysob = np.linspace(0, len(dcOb), len(dcOb))

fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(daysob, dcOb, 'y', alpha=0.5, lw=2, label='Observed Daily Cases')
ax.plot(dayssi, IDD, 'b', alpha=0.5, lw=2, label='Simulated Daily Cases')
ax.set_title('Observed vs Simulated Daily Cases 1/9/20 - 21/9/20')
ax.set_xlabel('Number of days since 1st September 2020')
ax.set_ylabel('Number of People')
ax.set_ylim(0,8000)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
