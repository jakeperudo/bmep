import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyabc
from pyabc.visualization import plot_kde_2d, plot_data_callback, plot_kde_1d_highlevel
import os
import tempfile



db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test2.db"))

h = pyabc.History(db_path)

df, w = h.get_distribution()

p={}
for x in range(0,15):
    p["{}".format(df.columns[x])] = [np.average(df[df.columns[x]]), np.percentile(df[df.columns[x]], 20, axis=0), np.percentile(df[df.columns[x]], 80, axis=0)]

for key, value in p.items():
    print(key, ' : ', value)
