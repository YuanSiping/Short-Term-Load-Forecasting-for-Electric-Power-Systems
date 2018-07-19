# 单变量负荷预测：以前24小时负荷数据预测下一小时负荷
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from tools import statistics
import math
import time
import operator as op
from matplotlib.font_manager import FontProperties
from tools import statistics

# define a function to convert a vector of time series into a 2D matrix 定义将时间序列向量转换为二维矩阵的函数
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

# load raw data 加载原始数据
df_raw = pd.read_csv('../data/ENTSO-E/load.csv', header=0, usecols=[0,1])
# numpy array
df_raw_array = df_raw.values
# daily load 加载日负载数据
list_hourly_load = [df_raw_array[i,1]/1000 for i in range(0, len(df_raw))]
print ("Data shape of list_hourly_load: ", np.shape(list_hourly_load))
# 异常值处理
k = 0
for j in range(0, len(list_hourly_load)):
    if(abs(list_hourly_load[j]-list_hourly_load[j-1])>2 and abs(list_hourly_load[j]-list_hourly_load[j+1])>2):
        k = k + 1
        list_hourly_load[j] = (list_hourly_load[j - 1] + list_hourly_load[j + 1]) / 2 + list_hourly_load[j - 24] - list_hourly_load[j - 24 - 1] / 2
    sum = 0
    num = 0
    for t in range(1,8):
        if(j - 24*t >= 0):
            num = num + 1
            sum = sum + list_hourly_load[j - 24*t]
        if(j + 24*t < len(list_hourly_load)):
            num = num + 1
            sum = sum + list_hourly_load[j + 24*t]
    sum = sum / num
    if(abs(list_hourly_load[j] - sum)>3):
        k = k + 1
        if(list_hourly_load[j] > sum): list_hourly_load[j] = sum + 3
        else: list_hourly_load[j] = sum - 3
# shift all data by mean 去均值
list_hourly_load = np.array(list_hourly_load)
shifted_value = list_hourly_load.mean()
list_hourly_load -= shifted_value
# the length of the sequnce for predicting the future value
sequence_length = 25
# convert the vector to a 2D matrix
matrix_load = convertSeriesToMatrix(list_hourly_load, sequence_length)
matrix_load = np.array(matrix_load)
print ("Data shape: ", matrix_load.shape)
# split dataset: 90% for training and 10% for testing 切分数据集
# train_row = int(round(0.9 * matrix_load.shape[0]))
train_row = matrix_load.shape[0] - 166*24
print('train:',train_row,'test:',166*24)
train_set = matrix_load[:train_row, :]
# random seed
np.random.seed(1234)
# shuffle the training set (but do not shuffle the test set)
np.random.shuffle(train_set)
# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1]
print(X_train[0],y_train[0])
# the test set
X_test = matrix_load[train_row:, :-1]
y_test = matrix_load[train_row:, -1]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# gbdt
# 模型超参数选择
# gbdt = GradientBoostingRegressor(subsample=1,
#                                  min_samples_split=2, min_samples_leaf=1, max_depth=3, alpha=0.9,
#                                  verbose=0)
# param_grid = {
#     'loss': ['ls', 'lad', 'huber'],
#     'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
#     'n_estimators': [100, 200, 400, 800, 1000],
#     'max_depth': [3, 4, 5, 6],
#     'alpha': [0.7, 0.8, 0.9]}
# gbm = GridSearchCV(gbdt, param_grid)
# gbm.fit(X_train, y_train[:,i])
# print('Best parameters found by grid search are:', gbm.best_params_)
#模型训练
gbdt = GradientBoostingRegressor(loss='ls', learning_rate=0.2, n_estimators=400, subsample=1,
                                    min_samples_split=2, min_samples_leaf=1, max_depth=3, alpha=0.7,
                                    verbose=0)
gbdt.fit(X_train, y_train)
# 展示特征的重要性分布
feature_importance = gbdt.feature_importances_
# get the predicted values
start = time.clock()
predicted_values = gbdt.predict(X_test)
print('预测耗时：', time.clock() - start, 's')
# 绘制特征的重要性分布
plt.figure()
plt.scatter(np.arange(1, len(feature_importance) + 1), feature_importance, c='r', zorder=10)
plt.plot(np.arange(1, len(feature_importance) + 1), feature_importance)
plt.xlabel('Feature index')
plt.ylabel('Feature importance')
plt.show()
# evaluation
mape = statistics.mape((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
print('MAPE is ', mape)
mae = statistics.mae((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
print('MAE is ', mae)
mse = statistics.meanSquareError((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
print('MSE is ', mse)
rmse = math.sqrt(mse)
print('RMSE is ', rmse)
nrmse = statistics.normRmse((y_test + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
print('NRMSE is ', nrmse)
# plot the results
fig = plt.figure()
plt.plot(y_test + shifted_value, label="$true$", c='green')
plt.plot(predicted_values + shifted_value, label="$predict$", c='red')
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e3)')
plt.legend()
plt.show()
fig.savefig('../result/gbrt_result.jpg', bbox_inches='tight')