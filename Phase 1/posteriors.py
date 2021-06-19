import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyabc
from pyabc.visualization import plot_histogram_1d, plot_kde_2d, plot_kde_1d, plot_data_callback, plot_kde_1d_highlevel, plot_kde_matrix
import os
import tempfile
from model import modelCov

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test2.db"))

h = pyabc.History(db_path)

df, w = h.get_distribution()

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
ax = plot_kde_1d(df,w, "lol", ax=ax, xname="Theta 0")
ax = fig.add_subplot(1, 2, 2)
ax = plot_kde_1d(df,w, "lol1", ax=ax, xname="Theta 1")

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
ax = plot_kde_1d(df,w, "lol3", ax=ax, xname="Theta 2")
ax = fig.add_subplot(1, 2, 2)
ax = plot_kde_1d(df,w, "lol4", ax=ax, xname="Theta 3")

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
ax = plot_kde_1d(df,w, "alpha1", ax=ax, xname="Alpha 1")
ax = fig.add_subplot(1, 2, 2)
ax = plot_kde_1d(df,w, "alpha2", ax=ax, xname="Alpha 2")

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
ax = plot_kde_1d(df,w, "beta1", ax=ax, xname="Beta 1")
ax = fig.add_subplot(1, 2, 2)
ax = plot_kde_1d(df,w, "beta2", ax=ax, xname="Beta 2")

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
ax = plot_kde_1d(df,w, "beta3", ax=ax, xname="Beta 3")
ax = fig.add_subplot(1, 2, 2)
ax = plot_kde_1d(df,w, "beta4", ax=ax, xname="Beta 4")

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
ax = plot_kde_1d(df,w, "rho1", ax=ax, xname="Rho 1")
ax = fig.add_subplot(1, 2, 2)
ax = plot_kde_1d(df,w, "rho2", ax=ax, xname="Rho 2")

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 2, 1)
ax = plot_kde_1d(df,w, "gamma1", ax=ax, xname="Gamma 1")
ax = fig.add_subplot(1, 2, 2)
ax = plot_kde_1d(df,w, "gamma2", ax=ax, xname="Gamma 2")

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot()
ax = plot_kde_1d(df,w, "epsilon", ax=ax, xname="Epsilon")

plt.show()
