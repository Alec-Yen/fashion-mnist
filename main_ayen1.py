import numpy as np
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


import zalando.utils.mnist_reader as mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train[0:60000]
y_train = y_train[0:60000]
X_test = X_test[0:10000]
y_test = y_test[0:10000]

y_train = y_train.reshape(y_train.size,1)
y_test = y_test.reshape(y_test.size,1)

print("X_train",X_train.shape)
print("y_train",y_train.shape)

tr = np.hstack((X_train,y_train))
te = np.hstack((X_test,y_test))


###########START TESTING##################################


ntr, nte = pp.normalize(tr,te)
ptr, pte, err = pp.pca(tr,te,0.1,0)
ftr, fte = pp.fld(tr,te)

print("Raw")
case1acc = cls.discriminant_accuracy(tr,te,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),1,verbose=1)
case2acc = cls.discriminant_accuracy(tr,te,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),2,verbose=1)
case3acc = cls.discriminant_accuracy(tr,te,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),3,verbose=1)
knnacc = cls.knn_accuracy(tr,te,10,0,0)

print("Case 1 Acc",case1acc)
print("Case 2 Acc",case2acc)
print("Case 3 Acc",case3acc)
print("KNN Acc",knnacc)



# print("PCA")
# # case1acc = cls.discriminant_accuracy(ptr,pte,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),1,verbose=0)
# # print("Case 1 Acc",case1acc)
# # case2acc = cls.discriminant_accuracy(ptr,pte,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),2,verbose=0)
# # print("Case 2 Acc",case2acc)
# # case3acc = cls.discriminant_accuracy(ptr,pte,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),3,verbose=0)
# # print("Case 3 Acc",case3acc)
# # knnacc = cls.knn_accuracy(ptr,pte,10,0,0)
# # print("KNN Acc",knnacc)
# #
# # print("FLD")
# # case1acc = cls.discriminant_accuracy(ftr,fte,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),1,verbose=0)
# # print("Case 1 Acc",case1acc)
# # case2acc = cls.discriminant_accuracy(ftr,fte,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),2,verbose=0)
# # print("Case 2 Acc",case2acc)
# # case3acc = cls.discriminant_accuracy(ftr,fte,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),3,verbose=0)
# # print("Case 3 Acc",case3acc)
# # knnacc = cls.knn_accuracy(ftr,fte,10,0,0)
# # print("KNN Acc",knnacc)
