import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_csv('../data/demanddata501125.csv')
# data = data.values
# time = data[:,0]
# L_dem = data[:,13]
# T_dem = data[:,14]
# S_dem = data[:,15]
# F_dem = data[:,16]
# plt.subplot(4,1,1)
# plt.plot(L_dem)
# plt.xlabel('Time')
# plt.ylabel('L_dem')
# plt.subplot(4,1,2)
# plt.plot(T_dem)
# plt.xlabel('Time')
# plt.ylabel('T_dem')
# plt.subplot(4,1,3)
# plt.plot(S_dem)
# plt.xlabel('Time')
# plt.ylabel('S_dem')
# plt.subplot(4,1,4)
# plt.plot(F_dem)
# plt.xlabel('Time')
# plt.ylabel('F_dem')
# plt.show()

data = pd.read_csv('../data/load_30s_0504_0715.csv')
data = data.values
T_dem = data[:,1]/10
for i in range(1,np.shape(data)[0]-1):
    if(abs(T_dem[i]-T_dem[i-1])>50 or abs(T_dem[i]-T_dem[i+1])>50):
        T_dem[i]=(T_dem[i-1]+T_dem[i+1])/2
for i in range(2880, np.shape(data)[0] - 2880):
    temp=(T_dem[i-2880]+T_dem[i+2880])/2
    if(abs(T_dem[i]-temp)>30):
        if(T_dem[i]>temp):
            T_dem[i] = T_dem[i] - 30
        if(T_dem[i]<temp):
            T_dem[i] = T_dem[i] + 30
plt.plot(T_dem)
plt.xlabel('Time')
plt.ylabel('Demand Load')
plt.show()