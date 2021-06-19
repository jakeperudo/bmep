#Without Delta
#G = []
#G.append(np.average(I[2:8]))
#G.append(np.average(I[6:12]))
#G.append(np.average(I[13:19]))
#G.append(np.average(I[18:24]))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyabc
from pyabc.visualization import plot_kde_2d, plot_data_callback, plot_kde_1d_highlevel
import os
import tempfile
from model import modelCov
db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test2.db"))

IDD_data = np.array([543, 554,#1-2
                   1040, 1020, 1058, 1083, 943, 695, 616,#3-9
                   1470, 1341, 1207, 1148, 1162, 694, 600,#10-16
                   1284, 1084, 1239, 1462, 1199, 826, 826,#17-23
                   1271, 1249, 1349, 1549, 1626, 1221, 1160,#24-30
                   1501#31
                   ])

D_data = np.array([41369, 41379,
                   41394, 41408, 41414, 41423, 41435, 41449, 41457,
                   41470, 41480, 41487, 41496, 41503, 41515, 41519,
                   41531, 41540, 41543, 41551, 41558, 41566, 41580,
                   41586, 41598, 41609, 41617, 41627, 41632, 41638,
                   41647
                   ])

G_data = np.array([28300, 24600, 28200, 27100])#3-9, 7-13, 14-20, 19-25

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
