import warnings
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

Load = ['../data/THESL/hourly_load_hoep_2016.csv']
labels = ['Load','HOEP']

# Possible number of states in Markov Model 状态数
STATE_SPACE = range(2,15)

# 时间尺度
K = 24
# 测试时间
NUM_TEST = 509
# 迭代次数
NUM_ITERS=1000

# Calculating Mean Absolute Percentage Error of predictions
def calc_mape(predicted_data, true_data):
    return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])

load = Load[0]
dataset = np.genfromtxt(load, delimiter=',')
#print(dataset.shape[1])
predicted = np.empty([0,dataset.shape[1]]) #存预测值
# likelihood_vect = np.empty([0,1])
# aic_vect = np.empty([0,1])
# bic_vect = np.empty([0,1])

'''
# 参数优化，选状态数
for states in STATE_SPACE:
    num_params = states ** 2 + states
    dirichlet_params_states = np.random.randint(1, 50, states)
    # model = hmm.GaussianHMM(n_components=states, covariance_type='full', startprob_prior=dirichlet_params_states, transmat_prior=dirichlet_params_states, tol=0.0001, n_iter=NUM_ITERS, init_params='mc')
    model = hmm.GaussianHMM(n_components=states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS)
    model.fit(dataset[:-NUM_TEST, :])
    if model.monitor_.iter == NUM_ITERS:
        print('需增加迭代次数')
        sys.exit(1)
    likelihood_vect = np.vstack((likelihood_vect, model.score(dataset)))
    aic_vect = np.vstack((aic_vect, -2 * model.score(dataset) + 2 * num_params))
    bic_vect = np.vstack((bic_vect, -2 * model.score(dataset) + num_params * np.log(dataset.shape[0])))

opt_states = np.argmin(bic_vect) + 2
print('最佳状态数目Optimum number of states are {}'.format(opt_states))
# 最佳状态数目Optimum number of states are 14
'''
opt_states = 14
for idx in range(1,NUM_TEST+1):
    print('idx:',idx)
    train_dataset = dataset[:-(idx + 1), :] #测试时间之前
    test_data = dataset[-idx, :]; #测试时间
    num_examples = train_dataset.shape[0]
    # model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', startprob_prior=dirichlet_params, transmat_prior=dirichlet_params, tol=0.0001, n_iter=NUM_ITERS, init_params='mc')
    if idx == 1:
        model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS,
                                init_params='stmc')
    else:
        # Retune the model by using the HMM paramters from the previous iterations as the prior
        model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS,
                                init_params='')
        model.transmat_ = transmat_retune_prior
        model.startprob_ = startprob_retune_prior
        model.means_ = means_retune_prior
        model.covars_ = covars_retune_prior

    model.fit(train_dataset)

    transmat_retune_prior = model.transmat_
    startprob_retune_prior = model.startprob_
    means_retune_prior = model.means_
    covars_retune_prior = model.covars_

    if model.monitor_.iter == NUM_ITERS:
        print('需增加迭代次数')
        sys.exit(1)

    past_likelihood = []
    curr_likelihood = model.score(train_dataset[-K:, :]) #预测时间前K时间序列 的 分数
    iters = 2;
    while iters < num_examples / K - 1:
        past_likelihood = np.append(past_likelihood, model.score(train_dataset[-(iters + K - 1):-iters, :])) #预测时间前所有K长度的时间序列 的 分数
        iters = iters + 1
    likelihood_diff_idx = np.argmin(np.absolute(past_likelihood - curr_likelihood))
    predicted_change = train_dataset[likelihood_diff_idx, :] - train_dataset[likelihood_diff_idx - 1, :] #相似状态时间-前一时间的 = 状态变化
    predicted = np.vstack((predicted, dataset[idx - 1, :] + predicted_change)) #预测时间前一时间 + 状态变化

np.savetxt('lhf_hmm_values_pred.csv', predicted, delimiter=',', fmt='%.2f')

mape = calc_mape(predicted, dataset[-NUM_TEST:, :])
print('MAPE is ', mape)

for i in range(2):
    plt.figure()
    plt.plot(range(NUM_TEST), predicted[:, i], 'k-', label='Predicted ' + labels[i]);
    plt.plot(range(NUM_TEST), dataset[-NUM_TEST:, i], 'r--', label='Actual ' + labels[i])
    plt.xlabel('Time steps')
    plt.ylabel(labels[i])
    plt.title('Result')
    plt.grid(True)
    plt.legend(loc='upper left')
plt.show()
