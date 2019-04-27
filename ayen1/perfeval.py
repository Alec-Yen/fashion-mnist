import numpy as np
import ayen1.classify_s as cls
import ayen1.preprocessing as pp
from sklearn.tree import DecisionTreeClassifier

"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition
perfeval.py

Purpose: Functions
This file includes function definitions regarding unsupervised classification.
"""

############################################################


"""
Purpose: Evaluate performance of a classifier using m-fold cross validation
Args:   groups: indexes of groups
        data: data set
        classifier_name - which classifier to use
            knn - k Nearest Neighbors
            dt - Decision Tree
            nn - 3-Layer Neural Network
        params - vary with classifier
            knn - [k, prior_prob, flag]
            dt - none
            nn - [num_classes, hidden_layer_units]
        cm - do you want to return matrix of accuracy
Returns:
"""

def mfold_cross_validation (group_indices, data, classifier_name,params=[],verbose=1,cm=0):
    # TODO: make groups an optional parameter
    np.random.seed(0)

    total_correct = 0
    total_accuracy = 0
    acc_arr = []
    confusion_matrix = np.zeros((group_indices.shape[0]))
    for i,g in enumerate(group_indices):
        mask = np.ones(data.shape[0],dtype=bool)
        mask[g] = 0
        tr = data[mask]
        te = data[g]
        tr,te = pp.normalize(tr,te) # normalize the data


        if classifier_name == "knn":
            acc = cls.knn_accuracy(tr,te,params[0],params[1],params[2])

        elif classifier_name == "dt":
            clf = DecisionTreeClassifier().fit(tr[:,:-1],tr[:,-1])
            Z = clf.predict(te[:,:-1])
            acc = np.sum(Z == te[:,-1])/len(Z)

        elif classifier_name == "nn":
            acc = cls.nn_3layer(tr,te,params[1],params[0],params[2])

        else:
            print("Invalid classifier")

        #        print("train",tr.shape)
        #        print("test",te.shape)
        total_correct += acc * te.shape[0]
        total_accuracy += acc
        confusion_matrix[i] = acc

        acc_arr = np.append(acc_arr,acc)

        if verbose == 1:
            print("m",i,"correct",acc*te.shape[0],"out of",te.shape[0])

#    return total_correct/data.shape[0]
    if cm==0:
        return total_accuracy/group_indices.shape[0], np.std(acc_arr)
    else:
        return total_accuracy/group_indices.shape[0], np.std(acc_arr), confusion_matrix


