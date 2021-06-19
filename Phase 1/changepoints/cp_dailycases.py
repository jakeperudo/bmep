import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import tempfile
import pyabc
from model import modelCov

db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test2.db"))

h = pyabc.History(db_path)

df, w = h.get_distribution()

p={}
for x in range(0,15):
    p["{}".format(df.columns[x])] = np.average(df[df.columns[x]])

dcOb = np.array([543, 554,#27-2
                   1040, 1020, 1058, 1083, 943, 695,
                   1470, 1341, 1207, 1148, 1162, 694, 600,#10-16
                   1284, 1084, 1239, 1462, 1199, 826, 826,#17-23
                   1271, 1249, 1349, 1549, 1626, 1221, 1160,#24-30
                   1501, #31
                   2249, 2292, 3098, 3039, 2544, 2443, 3859,
                   3302, 3331, 3593, 3289, 2666, 2141, 3392,
                   3548, 4360, 4629, 4959, 4854, 5320, 5604,
                   6403, 7059, 7603, 7241, 6756, 6971, 9927,
                   10235, 12555
                   ])

days = len(dcOb)

N, S, E, IA, IAD, IS, ISD, D, R, I, IDD, G = modelCov(p,days)

testdata=[]
for i in range(len(IDD)):
    testdata.append(dcOb[i]-IDD[i])
testdata = np.array(testdata)

#RUPTURES PACKAGE
#Changepoint detection with the Pelt search method
model="rbf"
algo = rpt.Pelt(model=model).fit(testdata)
result = algo.predict(pen=3)
rpt.display(testdata, result, figsize=(8, 6))
plt.title('Change Point Detection: Daily Reported Cases 1/8/20 - 30/9/20')
plt.xlabel('Days Since 1st August 2020')
plt.ylabel('Difference Between Simulated and Observed Daily Cases')
plt.subplots_adjust(top=0.95, left=0.1, bottom=0.15)
plt.show()
