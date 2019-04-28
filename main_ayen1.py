import numpy as np
from multiprocessing.pool import ThreadPool
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition
Final Project
main_ayen1.py

Purpose: Main Program
Description
"""

# files with my function definitions
import ayen1.perfeval as pe
import ayen1.classify_s as cls
import ayen1.preprocessing as pp

NUM_CLASSES = 10
prior_arr = np.repeat(1.0/NUM_CLASSES,NUM_CLASSES)

def case_tests (tr,te):
    case1acc = cls.discriminant_accuracy(tr,te,prior_arr,1,verbose=0)
    print("Case 1 Acc",case1acc)
    case2acc = cls.discriminant_accuracy(tr,te,prior_arr,2,verbose=0)
    print("Case 2 Acc",case2acc)
    case3acc = cls.discriminant_accuracy(tr,te,prior_arr,3,verbose=0)
    print("Case 3 Acc",case3acc)


def knn_threads (tr, te, k, num):
    pool = ThreadPool(processes=num)
    res = list()
    per_thread = round(te.shape[0]/num)
    for i in range(num):
        res.append(pool.apply_async(cls.knn_accuracy, (tr,te[i*per_thread:(i+1)*per_thread],k,0,0,)))
    acc = np.zeros(num)
    for i in range(num):
        acc[i] = res[i].get()

    print(acc)
    print(np.average(acc))
    return np.average(acc)


##############LOAD DATA#################################

import zalando.utils.mnist_reader as mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


y_train = y_train.reshape(y_train.size,1)
y_test = y_test.reshape(y_test.size,1)

tr = np.hstack((X_train,y_train))
te = np.hstack((X_test,y_test))


#######################PREPROCESSING##################################

# raw           Case I
# norm          Case II
# pca           Case III
# norm + pca    KNN
# fld
# norm + fld


#ntr, nte = pp.normalize(tr,te)
ptr, pte, perr = pp.pca(tr,te,0.1,0)
ftr, fte = pp.fld(tr,te)
#nptr, npte, nperr = pp.pca(ntr,nte,0.1,0)
#nftr, nfte = pp.fld(ntr,nte)


##########################M-FOLD CROSS VALIDATION################################################

# split up into groups for 10-fold cross validation
np.random.seed(0)
indexes = np.arange(tr.shape[0])
np.random.shuffle(indexes)
m = 10
groups = []
per_group = int(indexes.shape[0]/m)
for i in range(0,m):
    groups.append(indexes[i*per_group:(i+1)*per_group])
groups = np.array(groups)

#c1_cv_acc, c1_cv_std = pe.mfold_cross_validation(groups,tr,"case2",params=[prior_arr])
#print(c1_cv_acc)


##############################CLASSIFIER TESTING#############################################


print("Raw")
#case_tests(tr,te)
# k_arr = np.arange(5,21,5)
# knn_acc_arr = np.zeros(k_arr.shape[0])
# for i,k in enumerate(k_arr):
#     knn_acc_arr[i] = knn_threads(tr, te, k, 10)
#
# print(knn_acc_arr)
#
# plt.plot(k_arr, knn_acc_arr)
# plt.ylabel("KNN Accuracy")
# plt.xlabel("k")
# plt.savefig("latex/figures/knn_acc_arr.png")
# plt.show()



print("PCA")
#case_tests(ptr,pte)
print("FLD")
#case_tests(ftr,fte)

