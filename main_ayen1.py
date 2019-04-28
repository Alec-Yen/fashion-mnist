import numpy as np
import threading
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

def case_tests (tr,te):
    case1acc = cls.discriminant_accuracy(tr,te,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),1,verbose=0)
    print("Case 1 Acc",case1acc)
    case2acc = cls.discriminant_accuracy(tr,te,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),2,verbose=0)
    print("Case 2 Acc",case2acc)
    case3acc = cls.discriminant_accuracy(tr,te,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),3,verbose=0)
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


##############LOAD DATA#################################

import zalando.utils.mnist_reader as mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


y_train = y_train.reshape(y_train.size,1)
y_test = y_test.reshape(y_test.size,1)

tr = np.hstack((X_train,y_train))
te = np.hstack((X_test,y_test))


###########START TESTING##################################


ntr, nte = pp.normalize(tr,te)
ptr, pte, err = pp.pca(tr,te,0.1,0)
ftr, fte = pp.fld(tr,te)


print("Raw")
case_tests(tr,te)
#knn_threads(tr,te,10,10)

print("PCA")
case_tests(ptr,pte)
print("FLD")
case_tests(ftr,fte)

