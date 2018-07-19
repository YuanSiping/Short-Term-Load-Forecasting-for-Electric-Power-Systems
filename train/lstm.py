# 单变量负荷预测：以前24小时负荷数据预测下一小时负荷
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from tools import statistics
import math

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
print(k)
plt.plot(list_hourly_load)
plt.show()
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
time_test = [df_raw_array[i,0] for i in range(train_row+23, len(df_raw))]
# print(time_test[0]) # 7/10/2016 19:00
# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element) 输入(样本数量，每个元素维数)形式
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# lstm
# build the model 序贯模型
model = Sequential()
# layer 1: LSTM
model.add(LSTM( input_dim=1, output_dim=50, return_sequences=True))
model.add(Dropout(0.2))
# layer 2: LSTM
model.add(LSTM(output_dim=100, return_sequences=False))
model.add(Dropout(0.2))
# layer 3: dense
# linear activation: a(x) = x
model.add(Dense(output_dim=1, activation='linear'))
# show model
model.summary()
# compile the model
model.compile(loss="mse", optimizer="rmsprop")
# train the model
model.fit(X_train, y_train, batch_size=512, nb_epoch=50, validation_split=0.05, verbose=2)
# save model
model.save('../model/lstm.h5')
# load model
# from keras.models import load_model
model = load_model('../model/lstm.h5')
# evaluate the result
test_mse = model.evaluate(X_test, y_test, verbose=2)
print ('\nThe MSE on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test)))
# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))
# evaluation
mape = statistics.mape((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('MAPE is ', mape)
mae = statistics.mae((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('MAE is ', mae)
mse = statistics.meanSquareError((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('MSE is ', mse)
rmse = math.sqrt(mse)
print('RMSE is ', rmse)
nrmse = statistics.normRmse((y_test+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('NRMSE is ', nrmse)
# plot the results
fig = plt.figure()
plt.plot(y_test + shifted_value, label="$true$", c='green')
plt.plot(predicted_values + shifted_value, label="$predict$", c='red')
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e3)')
plt.legend()
plt.show()
fig.savefig('../result/lstm_result.jpg', bbox_inches='tight')
# save the result into csv file
#np.savetxt('../result/lstm_time.csv', time_test, fmt="%s", header="time")
#np.savetxt('../result/lstm_pred.csv', (predicted_values+shifted_value)*1000, fmt="%.8f", header="predicted_values")
#np.savetxt('../result/lstm_true.csv', (y_test+shifted_value)*1000, fmt="%.8f", header="true_values")
