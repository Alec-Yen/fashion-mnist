
from cedwar45.PCA import PCA, PCA_k
from cedwar45.kMeans import kMeans
import numpy as np
import matplotlib.pyplot as plt
import zalando.utils.mnist_reader as mnist_reader
import random

from keras.datasets import mnist
#import keras.losses
#from keras.models import Sequential
#from keras.layers import Dense


np.random.seed(321);


fashion = True;#set if you want fashion-mnist

if fashion:
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

else: #use regular MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


c = 10; #10 classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

n = X_train.shape[0];
d = X_train.shape[1];


pX_train, pX_test = PCA_k(X_train, X_test, 2);

idx=random.sample(range(n),1000)

plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = y_train);
plt.title("Original Label")
#plt.show()

#Use k-means on PCA
km_X, winners, km_MD = kMeans(pX_train, k=c, d=2, seed=123);

plt.figure()
plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = winners);
plt.title("k-means Clusters on PCA")
#plt.show()


#Use k-means on PCA
km_X, winners, km_MD = kMeans(X_train, k=c, d=X_train.shape[1], seed=123);

plt.figure()
plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = winners);
plt.title("k-means Clusters on normalized")
plt.show()










