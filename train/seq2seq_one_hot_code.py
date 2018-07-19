# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pandas import DataFrame
from pandas import concat
from numpy import argmax
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from keras.models import load_model
from tools import statistics
import math

# convert time series into supervised learning problem
# data: Sequence of observations as a list or 2D NumPy array. Required.
# n_in: Number of lag observations as input (X). Values may be between [1..len(data)] Optional. Defaults to 1.
# n_out: Number of observations as output (y). Values may be between [0..len(data)-1]. Optional. Defaults to 1.
# dropnan: Boolean whether or not to drop rows with NaN values. Optional. Defaults to True.
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):  # cols的list长度就是n_in+n_out
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:  # 拿序列中最开始的几个数做样本，他们的t-n值肯定是nan，将这些样本踢掉
        agg.dropna(inplace=True)  # 丢弃掉的行数是n_lags+n_seq-1，减1是因为提供的序列本身也算作被预测
    agg = agg.applymap(lambda x: np.int32(x))
    return agg

# convert data to strings
def to_string(X, y, n_numbers, largest):
    max_length = 3
    Xstr = []
    for pattern in X:
        element_list=[]
        for element in pattern:
            strp =str(element)
            strp=''.join([' ' for _ in range(max_length-len(strp))])+strp
            element_list.append(strp)
        element_ensem=','.join([aa for aa in element_list])
        Xstr.append(element_ensem)
    ystr=[]
    for pattern in y:
        element_list=[]
        for element in pattern:
            strp =str(element)
            strp=''.join([' ' for _ in range(max_length-len(strp))])+strp
            element_list.append(strp)
        element_ensem=','.join([aa for aa in element_list])
        ystr.append(element_ensem)
    return Xstr,ystr

def one_hot_encode(X, series_min,series_max,n_unique):
    gap=(series_max-series_min)/n_unique
    Xenc=[]
    for sequence in X:
        new_index_ensem=[]
        for value in sequence:
            new_index=(value-series_min)/gap
            if value == 18544:
                new_index = new_index-0.1
            new_index_ensem.append(int(new_index))
        encoding=[]
        if value == 18544:
            print(new_index_ensem, new_index, value, series_max, series_min, gap)
        for index in new_index_ensem:
            vector=[0 for _ in range(n_unique)]
            vector[index]=1
            encoding.append(vector)
        Xenc.append(encoding)
    return np.array(Xenc)

# decode a one hot encoded string
def one_hot_decode(y,series_min,series_max,n_unique):
    gap=(series_max-series_min)/n_unique
    y_dec=[]
    for encoded_seq in y:
        decoded_seq=[argmax(vector) for vector in encoded_seq] # tf.argmax(vector, 1):返回的是vector中的最大值的索引号
        decoded_seq=np.array(decoded_seq)
        decoded_seq_tran=list(decoded_seq*gap+series_min)
        y_dec.append(decoded_seq_tran)
    return y_dec

if __name__=='__main__':
    # load dataset
    dataset = read_csv('../data/ENTSO-E/load.csv', header=0, index_col=0, usecols=[0,1])
    series = dataset.values
    list_hourly_load = [series[i] / 1000 for i in range(0, len(series))]
    print ("Data shape of list_hourly_load: ", np.shape(list_hourly_load))
    # 异常值处理
    k = 0
    for j in range(0, len(list_hourly_load)):
        if (abs(list_hourly_load[j] - list_hourly_load[j - 1]) > 2 and abs(
                    list_hourly_load[j] - list_hourly_load[j + 1]) > 2):
            k = k + 1
            list_hourly_load[j] = (list_hourly_load[j - 1] + list_hourly_load[j + 1]) / 2 + list_hourly_load[j - 24] - \
                                  list_hourly_load[j - 24 - 1] / 2
        sum = 0
        num = 0
        for t in range(1, 8):
            if (j - 24 * t >= 0):
                num = num + 1
                sum = sum + list_hourly_load[j - 24 * t]
            if (j + 24 * t < len(list_hourly_load)):
                num = num + 1
                sum = sum + list_hourly_load[j + 24 * t]
        sum = sum / num
        if (abs(list_hourly_load[j] - sum) > 3):
            k = k + 1
            if (list_hourly_load[j] > sum):
                list_hourly_load[j] = sum + 3
            else:
                list_hourly_load[j] = sum - 3
    print(k)
    list_hourly_load = [list_hourly_load[i] * 1000 for i in range(0, len(series))]
    #plt.plot(list_hourly_load)
    #plt.show()
    # shift all data by mean 去均值
    series = np.array(list_hourly_load)
    shifted_value = series.mean()
    series -= shifted_value
    # preprocerssing
    series_min=min(series)
    series_max=max(series)
    print(series_min,series_max)
    series=series.reshape(-1,1)
    # 输入观测次数
    n_in=24
    # 输出观测次数
    n_out=1
    # 将时间序列转换为监督学习问题
    supervised_data=series_to_supervised(series,n_in,n_out)
    supervised_data=supervised_data.values
    # 划分数据集
    # n_test=np.int32(0.1*round(supervised_data.shape[0]))
    n_test=166*24
    train, test = supervised_data[0:-n_test], supervised_data[-n_test:]
    print(train.shape,test.shape) # (18662, 25) (2073, 25)
    # trainset
    n_unique = 300
    X_train, y_train = train[:, :n_in], train[:, n_in:]
    [X_train,y_train]=map(lambda a:list(a),[X_train,y_train])
    X_train=map(lambda a:list(a),X_train)
    y_train=map(lambda a:list(a),y_train)
    X_train=one_hot_encode(X_train,series_min,series_max,n_unique)
    y_train=one_hot_encode(y_train,series_min,series_max,n_unique)
    #testset
    X_test, y_test = test[:, :n_in], test[:, n_in:]
    [X_test, y_test] = map(lambda a: list(a), [X_test, y_test])
    X_test = map(lambda a: list(a), X_test)
    y_test = map(lambda a: list(a), y_test)
    X_test = one_hot_encode(X_test, series_min, series_max,n_unique)
    y_test = one_hot_encode(y_test, series_min, series_max,n_unique)
    # define LSTM configuration
    encoded_length = n_unique # n_unique = encoded_length
    n_in_seq_length=len(list(X_train[0]))
    n_out_seq_length=len(list(y_train[0]))
    # print(n_in_seq_length,n_out_seq_length) # 24 1
    print(np.shape(X_train),np.shape(y_train))
    # create LSTM
    model = Sequential()
    model.add(LSTM(150, batch_input_shape=(None,n_in_seq_length,encoded_length)))  #encoder 150即隐含层节点数 = 输出维度，encoded_length即输入维度，n_in_seq_length即输入步长
    model.add(Dropout(0.2))
    model.add(RepeatVector(n_out_seq_length))
    model.add(LSTM(150, return_sequences=True))  #decoder
    model.add(Dropout(0.2))
    model.add(LSTM(150, return_sequences=True))  #decoder
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(encoded_length, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # show model
    print(model.summary())
    # train LSTM
    history=model.fit(X_train, y_train, epochs=50, batch_size=50, validation_split=0.05, shuffle=False, verbose=2)
    # save model
    model.save('../model/seq2seq_code.h5')
    model = load_model('../model/seq2seq_code.h5')
    # evaluate on some new patterns
    result = model.predict(X_test, batch_size=50, verbose=0)
    # calculate error,evaluate the result
    predicted = one_hot_decode(result,series_min,series_max,n_unique)
    # plot the results
    fig = plt.figure()
    plt.plot(test[:, n_in:] + shifted_value, label="$true$", c='green')
    plt.plot(predicted + shifted_value, label="$predict$", c='red')
    plt.xlabel('Hour')
    plt.ylabel('Electricity load')
    plt.legend()
    plt.show()
    fig.savefig('../result/seq2seq_code_result.jpg', bbox_inches='tight')
    # evaluation
    mape = statistics.mape(test[:, n_in:] + shifted_value, predicted + shifted_value)
    print('MAPE is ', mape)
    nrmse = statistics.normRmse(test[:, n_in:] + shifted_value, predicted + shifted_value)
    print('NRMSE is ', nrmse)
    mae = statistics.mae(test[:, n_in:] + shifted_value, predicted + shifted_value)
    print('MAE is ', mae)
    mse = statistics.mse(test[:, n_in:] + shifted_value, predicted + shifted_value)
    print('MSE is ', mse)
    rmse = sqrt(mse)
    print('RMSE is ', rmse)