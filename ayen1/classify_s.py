import numpy as np
import time
from scipy.stats import mode
from scipy.spatial import distance
import operator
import ayen1.preprocessing as pp
from progress.bar import Bar

import keras.losses
from keras.models import Sequential
from keras.layers import Dense

"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition
classify_s.py

Purpose: Functions
This file includes function definitions regarding supervised classification, including
discriminant classifiers (Case I, II, and III) and non-parametric k-nearest neighbors.
Implementations exist to test these classifiers over different prior probabilities
and different k's for KNN.
"""

############################################################

"""
Purpose:
    Discriminant classifier, supervised learning
    Calculates maximum posterior probability (MPP) and classifies test sample
    Assumes Gaussian distribution
Args:
    x: feature column vector, not including class label
    mu: array of mean column vectors for each class
    cov: array of covariance matrices for each class
    prior: array of prior probabilities for each class
    case: flag for discriminant function (1, 2, or 3)
        case=1: Minimum Euclidean Distance. Assumes each class has independent features with same variance.
        case=2: Minimum Mahalanobis Distance. Assumes covariance matrices for all classes identical, but not a scalar of identity matrix. We pick the average.
        case=3: Most General Case. THe covariance matrices are not equal to each other.
Returns:
    index of which class had the higher MPP
"""
def mpp(x, mu, cov, prior, case, cov_av, inv_cov_av, cov_norm_arr, inv_cov_arr):
    c = cov.shape[0]  # how many classes (categories)
    g = np.zeros(c)

    for i in range(c):
        if case == 1:  # Case 1, using average of variance
            g[i] = -1 / (2 * cov_av[0][0]) * (x - mu[i]).T.dot(x - mu[i]) + np.log(prior[i])
        if case == 2:  # Case 2, using average of covariance matrices
            g[i] = -1 / 2 * (x-mu[i]).T.dot(inv_cov_av).dot(x - mu[i]) + np.log(prior[i])
        if case == 3:
            g[i] = -1/2*np.log(cov_norm_arr[i]) - 1/2*(x-mu[i]).T.dot(inv_cov_arr[i]).dot(x-mu[i]) + np.log(prior[i])
            #g[i] = -1 / 2 * np.log(np.linalg.norm(cov[i])) - 1 / 2 * (x-mu[i]).T.dot(inv_cov_av).dot(x - mu[i]) + np.log(prior[i])
    return np.argmax(g)  # return index with max value


"""
Purpose:
    Discriminant classifier, supervised learning
    Calculate accuracy, sensitivity, specificity, TP, TN, FP, FN
    Assumes Gaussian distribution
Args:
    tr: training set, including class label
    te: testing set, including class label
    prior: array of prior probabilities, one for each class
    case: flag for discriminant function (1, 2, or 3). See mpp for details
Returns:
    accuracy, sensitivity, specificity, TP, TN, FP, FN
"""
def discriminant_accuracy(tr, te, prior, case,verbose=0):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    correct = 0
    dims = np.max(tr[:,-1])+1
    mu,cov = pp.get_params_arr(pp.return_multiclass_as_array(tr,dims)) # get parameters from training data
    cov_av = np.mean(cov,axis=0)
    inv_cov_av = np.linalg.pinv(cov_av)
    cov_norm_arr = []
    inv_cov_arr = []
    for i in range(dims):
        cov_norm_arr.append(np.linalg.norm(cov[i]))
        inv_cov_arr.append(np.linalg.pinv(cov[i]))
#    print(cov_norm_arr)
#    print(inv_cov_arr)

    print(te.shape[0])
    for i,test_sample in enumerate(te): # iterate through test data
        x_test = test_sample[0:-1].reshape(-1, 1)  # reshape test features to column vector
        if verbose == 1:
            print("Sample",i)

        if mpp(x_test, mu, cov, prior, case, cov_av, inv_cov_av, cov_norm_arr, inv_cov_arr) == test_sample[-1]:
            correct += 1
            if test_sample[-1]==0:
                TN += 1
            else:
                TP += 1
        else:
            if test_sample[-1]==0:
                FP += 1
            else:
                FN += 1

    # return accuracy depending on if it is a binary classification problem
    if tr.shape[0] > 2:
        return correct/te.shape[0]
    else:
        return (TN+TP)/te.shape[0], TP/(TP+FN), TN/(TN+FP), TP, TN, FP, FN

"""
Purpose:
    Discriminant classifier, supervised learning
    Calculate accuracy, sensitivity, specificity for different prior probabilities
    Assumes Gaussian distribution
Args:
    tr: training set, including class label
    te: testing set, including class label
    prior_range: array of a range of prior probabilities, only defined for one class (other is the complement)
Returns:
    accuracy array, sensitivity array, specificity array
"""
def disc_acc_per_prior (tr, te, prior_range):
    print("Calculating discriminant accuracy per prior...")
    acc_vec = np.zeros((3, prior_range.shape[0]))  # accuracies for different P1
    spec_vec = np.zeros((3, prior_range.shape[0]))  # accuracies for different P1
    sens_vec = np.zeros((3, prior_range.shape[0]))  # accuracies for different P1
    for index, P_it in enumerate(prior_range):
        P_tmp = np.array((P_it, 1 - P_it))
        acc_vec[0,index], sens_vec[0, index], spec_vec[0,index] = discriminant_accuracy(tr, te, P_tmp, 1)[0:3]
        acc_vec[1,index], sens_vec[1, index], spec_vec[1,index] = discriminant_accuracy(tr, te, P_tmp, 2)[0:3]
        acc_vec[2,index], sens_vec[2, index], spec_vec[2,index] = discriminant_accuracy(tr, te, P_tmp, 3)[0:3]
    case1_acc = acc_vec[0]
    case2_acc = acc_vec[1]
    case3_acc = acc_vec[2]
    case1_sens = sens_vec[0]
    case2_sens = sens_vec[1]
    case3_sens = sens_vec[2]
    case1_spec = spec_vec[0]
    case2_spec = spec_vec[1]
    case3_spec = spec_vec[2]
    return case1_acc,case1_sens,case1_spec,case2_acc,case2_sens,case2_spec,case3_acc,case3_sens,case3_spec

"""
Purpose:
    Discriminant classifier, supervised learning
    Calculate TPR, FPR for ROC curve
    Assumes Gaussian distribution
Args:
    tr: training set
    te: testing set
    prior_range: array of a range of prior probabilities, only defined for one class (other is the complement)
Returns:
    true positive rate, false positive rate
"""
def disc_roc (tr, te, prior_range):
    print("Calculating discriminant ROC...")
    case1_tpr = np.zeros(prior_range.shape[0])
    case1_fpr = np.zeros(prior_range.shape[0])
    case2_tpr = np.zeros(prior_range.shape[0])
    case2_fpr = np.zeros(prior_range.shape[0])
    case3_tpr = np.zeros(prior_range.shape[0])
    case3_fpr = np.zeros(prior_range.shape[0])
    for index, prior_it in enumerate(prior_range):
        prior = np.array((prior_it, 1 - prior_it))
        TP,TN,FP,FN  = discriminant_accuracy(tr, te, prior, 1)[3:7]
        case1_tpr[index] = TP/(TP+FN)
        case1_fpr[index] = 1-TN/(TN+FP)
        TP,TN,FP,FN  = discriminant_accuracy(tr, te, prior, 2)[3:7]
        case2_tpr[index] = TP/(TP+FN)
        case2_fpr[index] = 1-TN/(TN+FP)
        TP,TN,FP,FN  = discriminant_accuracy(tr, te, prior, 3)[3:7]
        case3_tpr[index] = TP/(TP+FN)
        case3_fpr[index] = 1-TN/(TN+FP)
    return case1_tpr,case1_fpr,case2_tpr,case2_fpr,case3_tpr,case3_fpr



"""
Purpose:
    KNN classifier, non-parametric learning
    Classifies test sample based on KNN
    Measures execution time for KNN
Args:
    tr: training set
    te: testing set
    target_k: how many neighbors
    desired_prior: needed to change the weighting of the votes
    flag: 0 or 1
        flag=0: use training set's prior probability
        flag=1: use given prior probability
Returns:
    accuracy, sensitivity, specificity, TP, TN, FP, FN, execution time
"""
def knn_accuracy (tr, te, target_k, desired_prior, flag, verbose=0):
    start_time = time.time()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    correct = 0

    if flag==1:
        tr_prior0 = pp.return_2class_as_array(tr)[0].shape[0] / tr.shape[0]
        tr_prior = np.array([tr_prior0, 1 - tr_prior0])
        ratio = desired_prior/tr_prior

    for i,test_sample in enumerate(te):
        if verbose == 1:
            print("Sample",i)
        test_sample = test_sample.reshape(1,-1) # reshape as 1-row
        dist_arr = distance.cdist(test_sample[:,:-1],tr[:,:-1]) # find distance
        indices = np.argsort(dist_arr)[0,0:target_k] # get k indices with largest values
        neighbors = tr[indices]
        labels = neighbors[:,-1]

        # if we are using prior probability, weight the votes
        if flag==1:
            unique, counts = np.unique(labels, return_counts=True)
            lookup = dict(zip(unique, counts))
            for key in lookup:
                lookup[key] *= ratio[int(key)]
            vote = max(lookup.items(), key = operator.itemgetter(1))[0]
        else:
            vote = (mode(labels)[0])

        # count up accuracy
        if vote == test_sample[0,-1]:
            correct += 1
            if test_sample[0,-1]==0:
                TN += 1
            else:
                TP += 1
        else:
            if test_sample[0,-1]==0:
                FP += 1
            else:
                FN += 1
    execution_time = time.time()-start_time

    # return depending on if it's binary or not
    if tr.shape[0] > 2:
        return correct/te.shape[0]
    else:
        return (TN+TP)/te.shape[0], TP/(TP+FN), TN/(TN+FP), TP, TN, FP, FN, execution_time

"""
Purpose:
    KNN classifier, non-parametric learning
    Gets accuracy, sensitivity, and specificity for different k values
Args:
    tr: training set
    te: testing set
    k_range: k values to test
Returns:
    accuracy array, sensitivity array, specificity array, timing array
"""
def knn_acc_per_k(tr, te, k_range):
    print("Calculating kNN accuracy per k...")
    acc_vec = np.zeros(k_range.shape[0])
    sens_vec = np.zeros(k_range.shape[0])
    spec_vec = np.zeros(k_range.shape[0])
    timing = np.zeros(k_range.shape[0])

    for index,k in enumerate(k_range):
        acc_vec[index],sens_vec[index],spec_vec[index],_,_,_,_,timing[index]  = knn_accuracy(tr, te, k, 0, 0)
    return acc_vec,sens_vec,spec_vec,timing

"""
Purpose:
    KNN classifier, non-parametric learning
    Gets accuracy, sensitivity, and specificity for different prior probabilities
Args:
    tr: training set
    te: testing set
    prior_range: range of prior probabilities to test (only defined for one class, other is complement)
    target_k: fixed k value to test
Returns:
    accuracy array, sensitivity array, specificity array
"""
def knn_acc_per_prior(tr, te, prior_range, target_k):
    print("Calculating kNN accuracy per prior...")
    acc_vec = np.zeros(prior_range.shape[0])
    sens_vec = np.zeros(prior_range.shape[0])
    spec_vec = np.zeros(prior_range.shape[0])

    for index,prior_it in enumerate(prior_range):
        prior = np.array((prior_it, 1 - prior_it))
        acc_vec[index],sens_vec[index],spec_vec[index]  = knn_accuracy(tr, te, target_k, prior, 1)[0:3]
    return acc_vec,sens_vec,spec_vec

"""
Purpose:
    KNN classifier, non-parametric learning
    Get TPR and FPR for plotting ROC curve
Args:
    tr: training set
    te: testing set
    prior_range: range of prior probabilities to test (only defined for one class, other is complement)
    target_k: fixed k value to test
Returns:
    true positive rate, false positive rate
"""
def knn_roc (tr, te, prior_range, target_k):
    print("Calculating kNN ROC...")
    tpr_vec = np.zeros(prior_range.shape[0])
    tnr_vec = np.zeros(prior_range.shape[0])
    for index, prior_it in enumerate(prior_range):
        prior = np.array((prior_it, 1 - prior_it))
        TP,TN,FP,FN  = knn_accuracy(tr, te, target_k, prior, 1)[3:7]
        tpr_vec[index] = TP/(TP+FN)
        tnr_vec[index] = TN/(TN+FP)
    return tpr_vec,1-tnr_vec


"""
Purpose:
    3-Layer neural network for classification
Args:
    tr: training set
    te: testing set
    validation_prop: what percantage is used for validation
    n: hidden layer nodes
    num_classes: number of classes for one hot encoding
"""

def nn_3layer (tr, te, n, outputs,validation_prop):
    np.random.seed(0)
    x_train = tr[:,:-1]
    y_train = keras.utils.to_categorical(tr[:,-1], num_classes=outputs) # one hot
    x_test = te[:,:-1]
    y_test = keras.utils.to_categorical(te[:,-1], num_classes=outputs)


    model = Sequential()
    model.add(Dense(units=n, activation='relu', input_dim=tr.shape[1] - 1))
    model.add(Dense(units=outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100, batch_size=32,verbose=False,validation_split=validation_prop)
    loss_and_metrics = model.evaluate(x_test,y_test,batch_size=128,verbose=False)

    return loss_and_metrics[1], history # returns the accuracy


#            Z = model.predict(x_test,verbose=False)
#            Z = np.argmax(Z,axis=1)
#            print("Network Result\t",Z)
#            print("Ground Truth\t",te[:,-1].astype(int))
#            acc = np.sum(Z ==  te[:,-1])/len(Z)
