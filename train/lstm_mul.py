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
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from tools import statistics
import math
from keras.optimizers import SGD
from keras.models import load_model

# convert series to supervised learning 将时间序列转换为监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('../data/ENTSO-E/load.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
matrix_load = np.array(values)
matrix_load[:,0] = matrix_load[:,0]/1000
# 异常值处理
k = 0
for j in range(0, matrix_load.shape[0]):
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
print(k)
# print(matrix_load[1,0])
# shift all data by mean 去均值
shifted_value = matrix_load[:,0].mean()
matrix_load[:,0] -= shifted_value
for i in range(1, 9):
    matrix_load[:, i] = matrix_load[:, i] / 10
    shifted_valuei = matrix_load[:,i].mean()
    matrix_load[:,i] -= shifted_valuei
print(matrix_load.shape)
# frame as supervised learning
before = 24
end = 1
reframed = series_to_supervised(matrix_load, before, end)
print(reframed.shape)
# drop columns we don't want to predict
for i in range(76):
    reframed.drop(reframed.columns[[-1]], axis=1, inplace=True)
print(reframed.shape)
print(reframed.head())
# split into train and test sets
values = reframed.values
n_train_hours = values.shape[0] - 166*24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print('train:',n_train_hours,'test:',166*24)
# split into input and outputs
train_X, train_y = train[:, :-end], train[:, -end]
print(train_X.shape, train_y.shape)
test_X, test_y = test[:, :-end], test[:, -end]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], before, 77))
test_X = test_X.reshape((test_X.shape[0], before, 77))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# lstm
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True)) #时间步长=train_X.shape[2]=1
model.add(LSTM(100, return_sequences=False))
model.add(Dense(end))
model.summary()
# sgd = SGD( lr = 0.01, decay= 1e-6, momentum=0.9, nesterov=True )
# model.compile(loss='mse', optimizer=sgd)
model.compile(loss="mse", optimizer="rmsprop")
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=512, validation_split=0.05, verbose=1, shuffle=False)
# save model
model.save('../model/lstm_mul.h5')
# load model
model = load_model('../model/lstm_mul.h5')
# evaluate the result
test_mse = model.evaluate(test_X, test_y, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(test_y)))
rmse = math.sqrt(test_mse)
print('RMSE is ', rmse)
# get the predicted values
predicted_values = model.predict(test_X)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))
# evaluation
mape = statistics.mape((test_y+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('MAPE is ', mape)
mae = statistics.mae((test_y+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('MAE is ', mae)
mse = statistics.meanSquareError((test_y+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('MSE is ', mse)
rmse = math.sqrt(mse)
print('RMSE is ', rmse)
nrmse = statistics.normRmse((test_y+shifted_value)*1000,(predicted_values+shifted_value)*1000)
print('NRMSE is ', nrmse)
# plot the results
fig = plt.figure()
plt.plot(test_y + shifted_value, label="$true$", c='green')
plt.plot(predicted_values + shifted_value, label="$predict$", c='red')
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e3)')
plt.legend()
plt.show()
fig.savefig('../result/lstm_mul.jpg', bbox_inches='tight')
# save result
#np.savetxt('../result/lstm_mul_time.csv', time_test, fmt="%s", header="time")
#np.savetxt('../result/lstm_mul_pred.csv', (predicted_values+shifted_value)*1000, fmt="%.8f", header="predicted_values")
#np.savetxt('../result/lstm_mul_true.csv', (test_y+shifted_value)*1000, fmt="%.8f", header="true_values")