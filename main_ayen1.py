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

#X_test = X_test[0:150]
#y_test = y_test[0:150]

y_train = y_train.reshape(y_train.size,1)
y_test = y_test.reshape(y_test.size,1)

print("X_train",X_train.shape)
print("y_train",y_train.shape)

tr = np.hstack((X_train,y_train))
te = np.hstack((X_test,y_test))

#test = pp.return_multiclass_as_array(tr,10)
acc = cls.discriminant_accuracy(tr,te,np.repeat(1.0/NUM_CLASSES,NUM_CLASSES),3,verbose=1)
print(acc)