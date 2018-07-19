import math
from tools import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from scipy.spatial import distance
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import euclidean
from sklearn.cluster import SpectralClustering

# Performs K-Means Clustering on the ordered sequence
# of vectors x with parameter k, and returns a 2-tuple:
# First tuple value is list of centroids
# Second tuple value is vector x' of length equal to that
# of x, such that the ith
# value of x' is the cluster label for the ith example
# of the input x
def kMeansClustering(x,k):

    # Convert list into numpy format
    conv = np.asarray(x)

    # Compute the centroids
    centroids = kmeans(conv,k,iter=10)[0]

    # Relabel the x's
    labels = []
    for y in range(len(x)):
        minDist = float('inf')
        minLabel = -1
        for z in range(len(centroids)):
            e = euclidean(conv[y],centroids[z])
            if (e < minDist):
                minDist = e
                minLabel = z
        labels.append(minLabel)

    # Return the list of centroids and labels
    return (centroids,labels)

# Performs a weighted clustering on the examples in xTest
# Returns a 1-d vector of predictions
def predictClustering(clusters,clusterSets,xTest,metric):
    clustLabels = []
    simFunction = getDistLambda(metric)
    for x in range(len(xTest)):
        clustDex = -1
        clustDist = float('inf')
        for y in range(len(clusters)):
            dist = simFunction(clusters[y],xTest[x])
            if (dist < clustDist):
                clustDist = dist
                clustDex = y
        clustLabels.append(clustDex)
    predict = np.zeros(len(xTest))
    for x in range(len(xTest)):
        predict[x] = weightedClusterClass(xTest[x],clusterSets[clustLabels[x]],simFunction)
    return predict

# Performs a weighted cluster classification
def weightedClusterClass(xVector,examples,simFunction):
    pred = 0.0
    normalizer = 0.0
    ctr = 0
    for x in examples:
        similarity = 1.0/simFunction(xVector,x[0])
        pred += similarity*x[1]
        normalizer += similarity
        ctr += 1
    return (pred/normalizer)

def getDistLambda(metric):
    if (metric == "manhattan"):
        return lambda x,y : distance.cityblock(x,y)
    elif (metric == "cosine"):
        return lambda x,y : distance.cosine(x,y)
    else:
        return lambda x,y : distance.euclidean(x,y)

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
before = 1
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
train_X = train_X.reshape((train_X.shape[0], before*77))
test_X = test_X.reshape((test_X.shape[0], before*77))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# clustering
# Compute centroids and labels of data
ckmeans_365,lkmeans_365 = kMeansClustering(train_X,365)
c = [ckmeans_365]
l = [lkmeans_365]
algNames = ["true","k-means(365)"]
preds = []
preds.append(test_y)
for t in range(len(c)):
    # The centroids computed by the current clustering algorithm
    centroids = c[t]
    # The labels for the examples defined by the current clustering assignment
    labels = l[t]
    # Separate the training samples into cluster sets
    clusterSets = []
    # Time labels for the examples, separated into clusters
    timeLabels = []
    for x in range(len(centroids)):
        clusterSets.append([])
    for x in range(len(labels)):
        # Place the example into its cluster
        clusterSets[labels[x]].append((train_X[x], train_y[x]))
    # Compute predictions for each of the test examples
    predicted_values = predictClustering(centroids, clusterSets, test_X, "euclidean")
    mape = statistics.mape((test_y + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('MAPE is ', mape)
    mae = statistics.mae((test_y + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('MAE is ', mae)
    mse = statistics.meanSquareError((test_y + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('MSE is ', mse)
    rmse = math.sqrt(mse)
    print('RMSE is ', rmse)
    nrmse = statistics.normRmse((test_y + shifted_value) * 1000, (predicted_values + shifted_value) * 1000)
    print('NRMSE is ', nrmse)
    preds.append(predicted_values)
# show
fig = plt.figure()
colors = ["g","r","b","c","m","y","k","w"]
legendVars = []
for j in range(len(preds)):
    print(j)
    x, = plt.plot(preds[j]+shifted_value, color=colors[j])
    legendVars.append(x)
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e3)')
plt.legend(legendVars, algNames)
plt.show()
fig.savefig('../result/clustering_mul_result.jpg', bbox_inches='tight')