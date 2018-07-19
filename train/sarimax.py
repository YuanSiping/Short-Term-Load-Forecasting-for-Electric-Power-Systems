# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from tools import statistics

# Load the data
data = pd.read_csv('../data/ENTSO-E/load.csv',usecols=[0,1])#nrows =365*24
# A bit of pre-processing to make it nicer
data['date']=pd.to_datetime(data['date'])
data.set_index(['date'], inplace=True)
data.values[:,0] = data.values[:,0]/1000
print(data)
'''
# Plot the data
data.plot()
plt.ylabel('Load')
plt.xlabel('Hour')
plt.show()
# d
diff = data.diff(1)
diff.plot()
plt.ylabel('Load')
plt.xlabel('Hour')
plt.show()
'''
'''
# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
p = range(0, 4)
# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
'''
# 划分数据集
train_data = data['2014-12-31 23:00:00':'2016-12-01 21:00:00']
test_data = data['2016-12-01 22:00:00':'2017-05-16 21:00:00']
'''
# 参数优化
warnings.filterwarnings("ignore") # specify to ignore warning messages
AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue
# order=SARIMAX_model[AIC.index(min(AIC))][0],seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1]
# print(SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1])
# (3, 0, 1) (3, 1, 1, 12)
'''
# fit model
mod = sm.tsa.statespace.SARIMAX(train_data,order=(3, 0, 1),seasonal_order=(3, 1, 1, 24),enforce_stationarity=False,enforce_invertibility=False)
model = mod.fit()
# predict data it has not seen before
pred2 = model.get_forecast('2017-05-16 21:00:00')
print(pred2)
'''
# show result of  predictions
ax = data.plot(figsize=(20, 16))
# pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
# pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Load')
plt.xlabel('Hour')
plt.legend()
plt.show()
'''
# show result of  predictions
# pr = pred2.predicted_mean['2016-07-10 19:00:00':'2016-07-31 23:00:00']+shifted_value
# ax = (data['2016-07-10 19:00:00':'2016-07-31 23:00:00']+shifted_value).plot(figsize=(20, 16))
pr = pred2.predicted_mean['2016-12-01 22:00:00':'2017-05-16 21:00:00']
ax = (data['2016-12-01 22:00:00':'2017-05-16 21:00:00']).plot(figsize=(20, 16))
# plot the results
fig = plt.figure()
plt.plot(ax, label="$true$", c='green')
plt.plot(pr, label="$predict$", c='red')
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e3)')
plt.legend()
plt.show()
fig.savefig('../result/sarimax.jpg', bbox_inches='tight')
# quantify the accuracy of the prediction
prediction = pred2.predicted_mean['2016-12-01 22:00:00':'2017-05-16 21:00:00'].values
# flatten nested list
truth = list(itertools.chain.from_iterable(test_data.values))
# evaluation
mape = np.mean(np.abs((truth - prediction) / truth))
print('mape is {:.5f}'.format(mape))
mae = statistics.mae(truth,prediction)
print('MAE is ', mae)
mse = statistics.meanSquareError(truth,prediction)
print('MSE is ', mse)
rmse = statistics.Rmse(mse)
print('RMSE is ', rmse)
nrmse = statistics.normRmse(truth,prediction)
print('NRMSE is ', nrmse)