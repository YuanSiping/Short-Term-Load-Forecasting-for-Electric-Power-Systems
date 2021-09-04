import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

data = pd.read_csv('../data/load_weather_30s_.csv', index_col=0)
data = data.values
data[:,0] = data[:,0] / 10
gy = data[:,0]
np.savetxt('../data.csv',data[:,0] )

fig=plot_acf(gy)
plt.tick_params(labelsize=17)
font2 = {'family' : 'Times New Roman','size' : 20,}
plt.xlabel('Lag',font2)
plt.ylabel('Autocorrelation',font2)
# fig.savefig('../data/a.eps', dpi=300000)
plt.show()



# font2 = {'family' : 'Times New Roman','size' : 20,}
# plt.xlabel('Lag',font2)
# plt.ylabel('Autocorrelation',font2)
# plt.show()
# plt.savefig('../data/a.eps', dpi=300)