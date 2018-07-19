# 单变量负荷预测
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
from keras import backend as K
from keras import regularizers
from sklearn.neural_network import BernoulliRBM
from keras.models import load_model
import pywt
print(pywt.families,pywt.wavelist('coif'))

# define a function to convert a vector of time series into a 2D matrix 定义将时间序列向量转换为二维矩阵的函数
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

def dwt(a):
    [ca, cd] = pywt.dwt(a,'haar')
    return ca,cd

def idwt(ca,cd):
    ori = pywt.idwt(ca,cd,'haar')
    return ori

def generateData(sample, outputnum):
    a = np.array(sample)
    mu = np.mean(a)
    #sigma_2 = np.var(a) / 2
    sigma_2 = np.var(a) / 24 #方差
    result = np.random.normal(loc = mu, scale = np.sqrt(sigma_2), size = outputnum) #正态分布
    #result = np.random.logistic(loc=mu, scale=np.sqrt(sigma_2), size=outputnum)  #Logistic分布
    # result = np.random.laplace(loc=mu, scale=np.sqrt(sigma_2), size=outputnum)  #拉普拉斯/双指数分布
    print('mu = %f\tsigma^2 = %f'%(mu,sigma_2))
    return mu,sigma_2,result

def drawResult(mu,sigma_2,result):
    plt.figure(figsize=(10,8),dpi=80)
    count, bins, ignored = plt.hist(result, 30, normed=True)
    plt.plot(bins, 1/(np.sqrt(2 * np.pi * sigma_2)) *np.exp( - (bins - mu)**2 / (2 * sigma_2) ),linewidth=2, color='r')

def dataset(matrix_load,train_row):
    matrix_load = np.array(matrix_load)
    print("Data shape: ", matrix_load.shape)
    # 划分数据集
    train_set = matrix_load[:train_row, :]
    # random seed
    np.random.seed(1234)
    # shuffle the training set (but do not shuffle the test set)
    np.random.shuffle(train_set)
    # the training set
    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    # the test set
    X_test = matrix_load[train_row:, :-1]
    y_test = matrix_load[train_row:, -1]
    # the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element) 输入(样本数量，每个元素维数)形式
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    print(np.shape(X_train), np.shape(X_test))
    return X_train,y_train,X_test,y_test

def  root_mean_squared_error(actual, pred):
    return K.sqrt(K.mean(K.square(pred - actual), axis=-1))

def train_ca_cd(type,X_train,y_train,X_test,y_test):
    input_layer = X_train
    hidden_layer = [250, 500, 200]
    weight_rbm = []
    bias_rbm = []
    for i in range(len(hidden_layer)):
        print("DBN Layer {0} Pre-training".format(i + 1))
        rbm = BernoulliRBM(n_components=hidden_layer[i], learning_rate=0.0005, batch_size=512, n_iter=200, verbose=2,
                           random_state=1)
        rbm.fit(input_layer)
        # size of weight matrix is [input_layer, hidden_layer]
        weight_rbm.append(rbm.components_.T)
        bias_rbm.append(rbm.intercept_hidden_)
        input_layer = rbm.transform(input_layer)
    print('Pre-training finish.', np.shape(weight_rbm[0]), np.shape(bias_rbm[0]))
    test_rms = 0
    result = []
    model = Sequential()
    print('Fine-tuning start.')
    for i in range(0, len(hidden_layer)):
        print('i:', i)
        if i == 0:
            model.add(Dense(hidden_layer[i], activation='sigmoid', input_dim=np.shape(X_train)[1]))
        elif i >= 1:
            model.add(Dense(hidden_layer[i], activation='sigmoid'))
        else:
            pass
        layer = model.layers[i]
        layer.set_weights([weight_rbm[i], bias_rbm[i]])
    # model.add(Dense(np.shape(yTrain)[1], activation='linear'))
    model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
    # sgd = SGD(lr=0.005, decay=0)
    model.compile(loss='mse', optimizer="rmsprop")  # sgd
    model.fit(X_train, y_train, batch_size=150, epochs=100, verbose=5)
    model.save('../model/dwt_dbn_'+type+'_100.h5')
    print('Fine-tuning finish.')
    return model

# load raw data 加载原始数据
df_raw = pd.read_csv('.\load.csv', header=0, usecols=[0,1])
# numpy array
df_raw_array = df_raw.values
# daily load 加载日负载数据
list_hourly_load = [df_raw_array[i,1]/1000 for i in range(1, len(df_raw))]
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
list_hourly_load = np.array(list_hourly_load)
shifted_value = list_hourly_load.mean()
list_hourly_load -= shifted_value
print(list_hourly_load[0])
# dwt
ca, cd = dwt(list_hourly_load)
ca = ca / 10
shifted_value_ca = ca.mean()
ca -= shifted_value_ca
print(ca[0],cd[0])
print('DWT finish.')
print(np.shape(ca),np.shape(cd))
# the length of the sequnce for predicting the future value
sequence_length = 25
# convert the vector to a 2D matrix
ca_matrix_load = convertSeriesToMatrix(ca, sequence_length)
cd_matrix_load = convertSeriesToMatrix(cd, sequence_length)
# split dataset: 90% for training and 10% for testing 切分数据集
# train_row = int(round(0.9 * matrix_load.shape[0]))
train_row = len(ca_matrix_load) - ( 166 * 24 / 2)
print('train:', train_row, 'test:', 166 * 24 / 2)
y_test_true = list_hourly_load[-166 * 24:]
time_test = [df_raw_array[i, 0] for i in range(len(df_raw) - 166 * 24, len(df_raw))]
# dataset
ca_X_train,ca_y_train,ca_X_test,ca_y_test = dataset(ca_matrix_load,train_row)
cd_X_train,cd_y_train,cd_X_test,cd_y_test = dataset(cd_matrix_load,train_row)
# train
# ca_model = train_ca_cd('ca',ca_X_train,ca_y_train,ca_X_test,ca_y_test)
# cd_model = train_ca_cd('cd',cd_X_train,cd_y_train,cd_X_test,cd_y_test)
# load model
ca_model = load_model('../model/dwt_dbn_ca.h5')
# cd_model = load_model('../model/dwt_dbn_cd.h5')
# evaluate the result
ca_test_mse = ca_model.evaluate(ca_X_test, ca_y_test, verbose=2)
print('\nThe MAE of %s on the test data set is %.3f over %d test samples.' % ('ca',ca_test_mse, len(ca_y_test)))
# cd_test_mse = ca_model.evaluate(cd_X_test, cd_y_test, verbose=2)
# print('\nThe MSE of %s on the test data set is %.3f over %d test samples.' % ('cd',cd_test_mse, len(cd_y_test)))

# get the predicted values
ca_pred = ca_model.predict(ca_X_test)[:,0]
print('Lowpass coefficient estimation finish.')
mu, sigma_2, cd_pred = generateData(cd[0:train_row], outputnum=len(cd)-24-train_row)
# cd_pred = cd_model.predict(cd_X_test)[:,0]
print('Highpass coefficient estimation finish.')
#print(np.shape(ca_pred),np.shape(cd_pred))
predicted_values = idwt((ca_pred + shifted_value_ca)* 10, cd_pred)
#predicted_values = idwt(ca_pred+ca_shifted_value, cd_pred+cd_shifted_value)
print('IDWT finish.')

# mape = statistics.mape([y_test_true[i]*1000 for i in range(0,len(y_test_true))],(predicted_values)*1000
print(len(y_test_true),len(predicted_values))
mape = statistics.mape([(y_test_true[i]+shifted_value)*1000 for i in range(0,len(y_test_true))],(predicted_values+shifted_value)*1000)
print('MAPE is ', mape)
mae = statistics.mae([(y_test_true[i]+shifted_value)*1000 for i in range(0,len(y_test_true))],(predicted_values+shifted_value)*1000)
print('MAE is ', mae)
mse = statistics.meanSquareError([(y_test_true[i]+shifted_value)*1000 for i in range(0,len(y_test_true))],(predicted_values+shifted_value)*1000)
print('MSE is ', mse)
rmse = math.sqrt(mse)
print('RMSE is ', rmse)
nrmse = statistics.normRmse([(y_test_true[i]+shifted_value)*1000 for i in range(0,len(y_test_true))],(predicted_values+shifted_value)*1000)
print('NRMSE is ', nrmse)

fig = plt.figure(figsize=(12, 9), dpi=100)
plt.subplot(2, 1, 1)
plt.plot(ca_y_test, label="$lowpass_real$", c='green')
plt.plot(ca_pred, label="$lowpass_prediction$", c='red')
plt.legend(['lowpass_real', 'lowpass_prediction'], loc='upper right')
plt.title('lowpass coefficient prediction result', fontsize=16)
plt.subplot(2, 1, 2)
# plt.plot(cd[train_row:], label="$highpass_real$", c='green')
plt.plot(cd_y_test, label="$highpass_real$", c='green')
plt.plot(cd_pred, label="$highpass_prediction$", c='red')
plt.legend(['highpass_real', 'highpass_prediction'], loc='upper right')
plt.title('highpass coefficient prediction result', fontsize=16)
# plt.show()
fig.savefig('../result/dwt_dbn_cA_cD.jpg', bbox_inches='tight')
# plot the results
fig = plt.figure()
plt.plot(y_test_true, label="$true$", c='green')
plt.plot(predicted_values, label="$predict$", c='red')
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e3)')
plt.legend()
# plt.show()
fig.savefig('../result/dwt_dbn_result.jpg', bbox_inches='tight')
# save the result into csv file
# np.savetxt('../result/dwt_dbn_time.csv', time_test, fmt="%s", header="time")
# np.savetxt('../result/dwt_dbn_values_pred.csv', (predicted_values)*1000, fmt="%.8f", header="predicted_values")
# np.savetxt('../result/dwt_dbn_values_true.csv', (y_test_true)*1000, fmt="%.8f", header="true_values")
