import math
from tools import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            e = euclidean(conv[y],centroids[z]) # 欧式距离
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
time_test = [df_raw_array[i,0] for i in range(train_row+23, len(df_raw))]
# clustering
# Compute centroids and labels of data
ckmeans_365,lkmeans_365 = kMeansClustering(X_train,365)
c = [ckmeans_365]
l = [lkmeans_365]
algNames = ["true","k-means(365)"]
preds = []
preds.append(y_test)
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
        clusterSets[labels[x]].append((X_train[x], y_train[x]))
    # Compute predictions for each of the test examples
    predicted_values = predictClustering(centroids, clusterSets, X_test, "euclidean")
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
fig.savefig('../result/clustering_result.jpg', bbox_inches='tight')