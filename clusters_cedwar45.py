
from cedwar45.PCA import PCA, PCA_k
from cedwar45.kMeans import kMeans
from cedwar45.WTA import WTA
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
    

dictionary = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

threeD=True

if not threeD:
    pX_train, pX_test = PCA_k(X_train, X_test, 2);
else:
    pX_train, pX_test = PCA_k(X_train, X_test, 3);
    from mpl_toolkits.mplot3d import Axes3D



    
#idx=random.sample(range(n),1000)

fig = plt.figure()
if not threeD:
    plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = y_train);
else:
    ax = Axes3D(fig)
    ax.scatter(pX_train[:,0], pX_train[:,1], pX_train[:,2], s = .25, c = y_train);
    ax.view_init(68, 8)
plt.title("Original Label Clusters")
#plt.show()

#Use k-means on PCA
km_X, km_winners, km_MD = kMeans(pX_train, k=c, seed=123);

fig = plt.figure()
if not threeD:
    plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = km_winners);
else:
    ax = Axes3D(fig)
    ax.scatter(pX_train[:,0], pX_train[:,1], pX_train[:,2], s = .25, c = km_winners);
    ax.view_init(68, 8)
plt.title("k-means Clusters from PCA")
#plt.show()


#Use k-means on raw
km_X, km_winners, km_MD = kMeans(X_train, k=c, seed=123);

fig = plt.figure()
if not threeD:
    plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = km_winners);
else:
    ax = Axes3D(fig)
    ax.scatter(pX_train[:,0], pX_train[:,1], pX_train[:,2], s = .25, c = km_winners);
    ax.view_init(68, 8)
plt.title("k-means Clusters from normalized")
#plt.show()


#Use WTA on PCA
WTA_X, WTA_winners, WTA_MD = WTA(pX_train, k=c, epsilon = .01, stop = 100, seed = 123)

fig = plt.figure()
if not threeD:
    plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = WTA_winners);
else:
    ax = Axes3D(fig)
    ax.scatter(pX_train[:,0], pX_train[:,1], pX_train[:,2], s = .25, c = WTA_winners);
    ax.view_init(68, 8)
plt.title("WTA Clusters from PCA")
#plt.show()


#Use WTA on raw
WTA_X, WTA_winners, WTA_MD = WTA(X_train, k=c, epsilon = .01, stop = 100, seed = 123)

fig = plt.figure()
if not threeD:
    plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = WTA_winners);
else:
    ax = Axes3D(fig)
    ax.scatter(pX_train[:,0], pX_train[:,1], pX_train[:,2], s = .25, c = WTA_winners);
    ax.view_init(68, 8)
plt.title("WTA Clusters from normalized")
plt.show()



#Test with k-means on FLD
pred = np.loadtxt("data/kmeans_predicted_fld.txt", dtype=int)


fig = plt.figure()
if not threeD:
    plt.scatter(pX_train[:,0], pX_train[:,1], s = .25, c = pred);
else:
    ax = Axes3D(fig)
    ax.scatter(pX_train[:,0], pX_train[:,1], pX_train[:,2], s = .25, c = pred);
    ax.view_init(68, 8)
plt.title("k-means Clusters from FLD")
plt.show()



