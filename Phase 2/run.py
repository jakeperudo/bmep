#Without Delta


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyabc
from pyabc.visualization import plot_kde_2d, plot_data_callback, plot_kde_1d_highlevel
import os
import tempfile
from model import modelCov
db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test1.db"))

IDD_data = np.array([2249, 2292, 3098, 3039, 2544, 2443, 3859,
                   3302, 3331, 3593, 3289, 2666, 2141, 3392,
                   3548, 4360, 4629, 4959, 4854, 5320, 5604
                   ])

D_data = np.array([41649, 41658, 41668, 41675, 41688, 41696, 41711,
                   41720, 41729, 41742, 41755, 41773, 41789, 41810,
                   41828, 41854, 41881, 41905, 41925, 41957,
                   41984])

G_data = np.array([59800, 103600])#3-9, 7-13, 14-20, 19-25

F_data = np.concatenate((IDD_data, D_data, G_data))

parameter_prior = pyabc.Distribution(
                lol=pyabc.RV("uniform", 0.5, 1),
                lol1=pyabc.RV("uniform", 0, 0.2),
                lol3=pyabc.RV("uniform", 0.2, 0.8),
                lol4=pyabc.RV("uniform", 0, 1),
                alpha1=pyabc.RV("beta", 4, 18),
                alpha2=pyabc.RV("uniform", 0.1, 0.3),
                beta1=pyabc.RV("uniform", 0, 1),
                beta2=pyabc.RV("uniform", 0, 1),
                beta3=pyabc.RV("uniform", 0, 1),
                beta4=pyabc.RV("uniform", 0, 1),
                rho1=pyabc.RV("uniform", 0, 1),
                rho2=pyabc.RV("uniform", 0, 1),
                gamma1=pyabc.RV("beta", 3, 18),
                gamma2=pyabc.RV("beta", 3, 27),
                epsilon=pyabc.RV("beta", 2, 24))

def distance(simulation, data):
    return np.sqrt((data["F"] - simulation["F"])**2).sum()

def model(pars):
    days = len(D_data)
    N, S, E, IA, IAD, IS, ISD, D, R, I, IDD, G = modelCov(pars,days)
    F = IDD + D + G
    return {'F': F}

abc = pyabc.ABCSMC(model, parameter_prior, distance_function=distance,
                   population_size=500)

abc.new(db_path, {'F': F_data})

h = abc.run(max_nr_populations=100)
