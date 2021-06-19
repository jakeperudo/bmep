import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import tempfile
import pyabc
from model import modelCov

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test1.db"))

h = pyabc.History(db_path)

df, w = h.get_distribution()

p={}
for x in range(0,15):
    p["{}".format(df.columns[x])] = np.average(df[df.columns[x]])

dOb = np.array([41649, 41658, 41668, 41675, 41688, 41696, 41711,
                   41720, 41729, 41742, 41755, 41773, 41789, 41810,
                   41828, 41854, 41881, 41905, 41925, 41957,
                   41984, 42023, 42079, 42114, 42146, 42185, 42232,
                   42286, 42355, 42392
                   ])

days = len(dOb)

N, S, E, IA, IAD, IS, ISD, D, R, I, IDD, G = modelCov(p,days)

testdata=[]
for i in range(len(D)):
    testdata.append(dOb[i]-D[i])
testdata = np.array(testdata)

#RUPTURES PACKAGE
#Changepoint detection with the Pelt search method
model="rbf"
algo = rpt.Pelt(model=model).fit(testdata)
result = algo.predict(pen=3)
rpt.display(testdata, result, figsize=(8, 6))
plt.title('Change Point Detection: Cumulative Deaths 1/8/20 - 30/9/20')
plt.xlabel('Days Since 1st August 2020')
plt.ylabel('Difference Between Simulated and Observed Cumulative Deaths')
plt.subplots_adjust(top=0.95, left=0.1, bottom=0.15)
plt.show()
