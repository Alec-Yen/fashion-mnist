



from __future__ import print_function
from cedwar45.mfold import mfold
import numpy as np
import matplotlib.pyplot as plt
import zalando.utils.mnist_reader as mnist_reader
from cedwar45.DTree import skLearn_tree_class
from cedwar45.PCA import PCA, PCA_k
from cedwar45.ConfusionMatrix import ConfusionMatrix
from ayen1.preprocessing import fld

from keras.datasets import mnist
import keras.losses
from keras.models import Sequential
from keras.layers import Dense


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

#Dimensionality Reduction
if False: 
    pX_train, pX_test = PCA(X_train, X_test, .1);
    print("PCA Done")
    fX_train, fX_test = fld(np.hstack((X_train, y_train.reshape(len(y_train), 1))), np.hstack((X_test, y_test.reshape(len(y_test), 1))));
    fX_train = fX_train[:,0:9];
    fX_test = fX_test[:,0:9];
    print("FLD Done")

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
if False: #skip

    #acc, std = mfold(X_grp, X_classes_grp, skLearn_tree_class);

    #print("sklearn decision tree 10-fold: \t", acc)
    
    
    #Test version
    predicted = skLearn_tree_class(X_train, X_test, y_train)
    CM = ConfusionMatrix(predicted, y_test, c);
    np.savetxt("data/DTree_predicted_raw.txt", predicted, "%d")
    np.savetxt("data/DTree_cm_raw.txt", CM, "%d");
    acc = accuracy(predicted, y_test)
    print("sklearn decision tree test: \t", acc)
    
    
    predicted = skLearn_tree_class(pX_train, pX_test, y_train)
    CM = ConfusionMatrix(predicted, y_test, c);
    np.savetxt("data/DTree_predicted_pca.txt", predicted, "%d")
    np.savetxt("data/DTree_cm_pca.txt", CM, "%d");
    acc = accuracy(predicted, y_test)
    print("PCA sklearn decision tree test: \t", acc)
    
    
    predicted = skLearn_tree_class(fX_train, fX_test, y_train)
    CM = ConfusionMatrix(predicted, y_test, c);
    np.savetxt("data/DTree_predicted_fld.txt", predicted, "%d")
    np.savetxt("data/DTree_cm_fld.txt", CM, "%d");
    acc = accuracy(predicted, y_test)
    print("FLD sklearn decision tree test: \t", acc)



np.random.seed(321)

#Multilayer Perceptron from sklearn
if False: #Skip MLP sklearn
    accs = np.array([]);
    stds = np.array([]);
    for h in [8]:#range(2,15+1):
        from sklearn.neural_network import MLPClassifier

        def skLearn_MLPClassifier(tr_features, te_features, tr_classes):

            #mlp = MLPClassifier(hidden_layer_sizes=(8), validation_fraction = 1/9, early_stopping=True, n_iter_no_change=5000, max_iter = 15000) 
            mlp = MLPClassifier(hidden_layer_sizes=(h), max_iter = 10000) 

            
            mlp.fit(tr_features, tr_classes)  
            predicted = mlp.predict(te_features);
            
            
            #plt.plot(np.array(range(len(mlp.loss_curve_)))/len(tr_classes), mlp.loss_curve_);
            #plt.xlabel("Epoch");
            #plt.ylabel("Loss");
            #plt.show();
            
            print("Done")
            
            return predicted;
            
            
        acc, std = mfold(X_grp, X_classes_grp, skLearn_MLPClassifier);
        
        accs = np.append(accs, acc);
        stds = np.append(stds, std);
        
        
        print("sklearn Multilayer Perceptron: ", h," \t", acc)
        
        

    plt.errorbar(list(range(2,15+1)), accs*100, stds*100, linestyle='None', marker='o')
    plt.xlabel('Hidden Nodes')
    plt.ylabel('Accuracy')
    plt.show()
    
    
    
np.random.seed(321)

#3 layer NN from Keras 
if False: #Skip MLP sklearn
    accs = np.array([]);
    stds = np.array([]);
    for h in [5,10,15]:#range(2,15+1):

        #using modified nn_3layer from ayen1
            
        def nn_3layer (tr_features, te_features, tr_classes, dim = d):  
            x_train = tr_features
            y_train = keras.utils.to_categorical(tr_classes, num_classes=c) # one hot
            x_test = te_features

            model = Sequential()
            model.add(Dense(d, input_dim=dim, activation='relu')) 
            model.add(Dense(h, activation='relu'))
            model.add(Dense(c, activation='softmax')) 
            model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])
                          
            cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
                          
            history = model.fit(x_train, y_train, epochs=100, batch_size=128,verbose=True,
                callbacks = [cb],
                #validation_data=(x_test, keras.utils.to_categorical(y_test, num_classes=c))
                validation_split = .1
                )
            #loss_and_metrics = model.evaluate(x_test,y_test,batch_size=128,verbose=False)

            #return loss_and_metrics[1], history # returns the accuracy
            
            
            predicted = model.predict(x_test,batch_size=128,verbose=False)
            predicted = np.argmax(predicted, axis = 1);
            #print(predicted.shape)
            
            
            #plt.plot(np.array(range(len(mlp.loss_curve_)))/len(tr_classes), mlp.loss_curve_);
            #plt.xlabel("Epoch");
            #plt.ylabel("Loss");
            #plt.show();
            
            
            return predicted;
            
        
        acc, std = mfold(X_grp, X_classes_grp, nn_3layer);
        
        accs = np.append(accs, acc);
        stds = np.append(stds, std);
        
        print("sklearn Multilayer Perceptron 10-fold: ", h," \t", acc)
        
        
        '''
        #Test version
        predicted = nn_3layer(X_train, X_test, y_train)
        CM = ConfusionMatrix(predicted, y_test, c);
        np.savetxt("data/k3NN_h"+str(h)+"_predicted_raw.txt", predicted, "%d")
        np.savetxt("data/k3NN_h"+str(h)+"_cm_raw.txt", CM, "%d");
        acc = accuracy(predicted, y_test)
        print("sklearn keras 3-layer NN test: ", h," \t", acc)
        
        
        predicted = nn_3layer(pX_train, pX_test, y_train, pX_train.shape[1])
        CM = ConfusionMatrix(predicted, y_test, c);
        np.savetxt("data/k3NN_h"+str(h)+"_predicted_pca.txt", predicted, "%d")
        np.savetxt("data/k3NN_h"+str(h)+"_cm_pca.txt", CM, "%d");
        acc = accuracy(predicted, y_test)
        print("PCA keras 3-layer NN test: ", h," \t", acc)
        
        
        predicted = nn_3layer(fX_train, fX_test, y_train, fX_train.shape[1])
        CM = ConfusionMatrix(predicted, y_test, c);
        np.savetxt("data/k3NN_h"+str(h)+"_predicted_fld.txt", predicted, "%d")
        np.savetxt("data/k3NN_h"+str(h)+"_cm_fld.txt", CM, "%d");
        acc = accuracy(predicted, y_test)
        print("FLD keras 3-layer NN test: ", h," \t", acc)
        '''
        
        
        

    #plt.errorbar(list(range(2,15+1)), accs*100, stds*100, linestyle='None', marker='o')
    #plt.xlabel('Hidden Nodes')
    #plt.ylabel('Accuracy')
    #plt.show()
    
    
    

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



classes=list(dictionary.values())

from cedwar45.ConfusionMatrix import plot_confusion_matrix

pred = np.loadtxt("data/knn_predicted_pca_k5_p1.txt", dtype=int)
plot_confusion_matrix(y_test, pred, np.array(classes))
plt.title("kNN (k=5, p=1) PCA Confusion Matrix")
plt.show()

