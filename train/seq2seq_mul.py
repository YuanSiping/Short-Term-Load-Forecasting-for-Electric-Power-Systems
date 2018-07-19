#多变量预测
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import numpy as np
import pandas as pd
from tools import statistics
import math
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.optimizers import SGD
from keras.models import load_model

# define a function to convert a vector of time series into a 2D matrix 定义将时间序列向量转换为二维矩阵的函数
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(np.shape(vectorSeries)[0]-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length,:])
    return matrix

# load raw data 加载原始数据
df_raw = pd.read_csv('../data/ENTSO-E/load.csv', header=0)
# numpy array
df_raw_array = df_raw.values
df_raw_array = df_raw_array[:,1:].astype('float32')
matrix_load = np.array(df_raw_array)
print ("Data shape: ", np.shape(matrix_load))
# 异常值处理
k = 0
matrix_load[:,0] = matrix_load[:,0] / 1000
for j in range(0, matrix_load.shape[0]-1):
	if(abs(matrix_load[j,0]-matrix_load[j-1,0])>2 and abs(matrix_load[j,0]-matrix_load[j+1,0])>2):
		k = k + 1
		matrix_load[j,0] = (matrix_load[j - 1,0] + matrix_load[j + 1,0]) / 2 + matrix_load[j - 24,0] - matrix_load[j - 24 - 1,0] / 2
	sum = 0
	num = 0
	for t in range(1,8):
		if(j - 24*t >= 0):
			num = num + 1
			sum = sum + matrix_load[j - 24*t,0]
		if((j + 24*t) < matrix_load.shape[0]):
			num = num + 1
			sum = sum + matrix_load[j + 24*t,0]
	sum = sum / num
	if(abs(matrix_load[j,0] - sum)>3):
		k = k + 1
		if(matrix_load[j,0] > sum): matrix_load[j,0] = sum + 3
		else: matrix_load[j,0] = sum - 3
# shift all data by mean 去均值
shifted_value = matrix_load[:,0].mean()
matrix_load[:,0] -= shifted_value
for i in range(1, 9):
    matrix_load[:, i] = matrix_load[:, i] / 10
    shifted_valuei = matrix_load[:, i].mean()
    matrix_load[:, i] -= shifted_valuei
# the length of the sequnce for predicting the future value
sequence_length = 25
# convert the vector to a 2D matrix
matrix_load = convertSeriesToMatrix(matrix_load, sequence_length)
matrix_load = np.array(matrix_load)
# 切分数据集
train_row = matrix_load.shape[0] - 166 * 24
train_set = matrix_load[:train_row, :, :]
test_set = matrix_load[train_row:, :, :]
np.random.seed(1234)
np.random.shuffle(train_set)
X_train = train_set[:,0:24,:]
y_train = train_set[:,24,0]
y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
X_test = test_set[:,0:24,:]
y_test = test_set[:,24,0]
print ("Data shape: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# create LSTM
model = Sequential()
model.add(LSTM(150, batch_input_shape=(None,X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(RepeatVector(1))
model.add(LSTM(150, return_sequences=True))  #decoder
model.add(Dropout(0.2))
model.add(LSTM(150, return_sequences=True))  #decoder
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(1, activation='linear')))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
# show model
print(model.summary())
# train LSTM
history=model.fit(X_train, y_train, epochs=50, batch_size=50, validation_split=0.05, shuffle=False, verbose=2)
# save model
model.save('../model/seq2seq_mul.h5')
model = load_model('../model/seq2seq_mul.h5')
# predict
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples, 1))
# evaluate on some new patterns
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
fig.savefig('../result/seq2seq_mu1_result.jpg', bbox_inches='tight')