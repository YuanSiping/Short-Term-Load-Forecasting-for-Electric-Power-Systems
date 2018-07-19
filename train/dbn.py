from __future__ import print_function
import math
import xlrd
from tools import statistics
#import visualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#from datagen import constructData
from keras import backend as K
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import regularizers
from sklearn.neural_network import BernoulliRBM

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
X_train = np.reshape(X_train, (np.shape(X_train)[0], np.shape(X_train)[1]))
X_test = np.reshape(X_test, (np.shape(X_test)[0], np.shape(X_test)[1]))
print(np.shape(X_train), np.shape(X_test))
print(np.shape(y_train), np.shape(y_test))
# dbn
input_layer = X_train
hidden_layer=[250,500,200]
weight_rbm = []
bias_rbm = []
for i in range(len(hidden_layer)):
    print("DBN Layer {0} Pre-training".format(i + 1))
    rbm = BernoulliRBM(n_components=hidden_layer[i],learning_rate=0.0005,batch_size=512,n_iter=200,verbose=2,random_state=1)
    rbm.fit(input_layer)
    # size of weight matrix is [input_layer, hidden_layer]
    weight_rbm.append(rbm.components_.T)
    bias_rbm.append(rbm.intercept_hidden_)
    input_layer = rbm.transform(input_layer)
print('Pre-training finish.',np.shape(weight_rbm[0]),np.shape(bias_rbm[0]))
test_rms = 0
result = []
model = Sequential()
print('Fine-tuning start.')
for i in range(0, len(hidden_layer)):
    print('i:',i)
    if i == 0:
        model.add(Dense(hidden_layer[i], activation='sigmoid',input_dim=np.shape(X_train)[1]))
    elif i >= 1:
        model.add(Dense(hidden_layer[i], activation='sigmoid'))
    else:
        pass
    layer = model.layers[i]
    layer.set_weights([weight_rbm[i], bias_rbm[i]])
# model.add(Dense(np.shape(yTrain)[1], activation='linear'))
model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
# sgd = SGD(lr=0.005, decay=0)
model.compile(loss='mse',optimizer="rmsprop")#sgd
model.fit(X_train, y_train, batch_size=150, epochs=100, verbose=5)
# save model
model.save('../model/dbn.h5')
print('Fine-tuning finish.')
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
fig.savefig('../result/dbn_result.jpg', bbox_inches='tight')
# save result
# np.savetxt('../result/dbn_pred.csv', result, fmt="%.8f",header="predicted_values")
# np.savetxt('../result/dbn_true.csv', yTest, fmt="%.8f",header="trued_values")
