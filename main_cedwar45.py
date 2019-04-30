



from __future__ import print_function
from keras.datasets import mnist
from cedwar45.mfold import mfold
import numpy as np
import zalando.utils.mnist_reader as mnist_reader


np.random.seed(321);


fashion = True;#set if you want fashion-mnist

if fashion:
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

else: #use regular MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

n = X_train.shape[0];

groups = [];
m = 10;
indices = np.random.permutation(np.array(range(0,n)));
for i in range(m):
    groups.append(indices[int(i/m * n):int((i+1)/m * n)]);

X_grp = [];
X_classes_grp = [];
for i in groups:
    X_grp.append(X_train[i]);
    X_classes_grp.append(y_train[i]);

def accuracy(predicted, te_classes):
    n_te = predicted.shape[0];
    
    rv = (sum(predicted==te_classes)/n_te); #total accuracy
    return rv
    
    
#Decision tree from sklearn
from sklearn import tree

#Test version

clf = tree.DecisionTreeClassifier();

clf.fit(X_train, y_train)  
predicted = clf.predict(X_test);
acc = accuracy(predicted, y_test);
print("sklearn decision tree test: \t", acc)

#M-fold
def skLearn_tree_class(tr_features, te_features, tr_classes):

    clf = tree.DecisionTreeClassifier();
    
    clf.fit(tr_features, tr_classes)  
    predicted = clf.predict(te_features);
    return predicted;
    

acc = mfold(X_grp, X_classes_grp, skLearn_tree_class);

print("sklearn decision tree 10-fold: \t", acc)



