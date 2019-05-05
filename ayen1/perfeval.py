import numpy as np
import ayen1.classify_s as cls
from sklearn.tree import DecisionTreeClassifier
import itertools

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
            case123 - [prior_arr]
            knn - [k, prior_prob, flag, distance_flag]
            knn_thread - [k, num_thread, distance_flag]
            dt - none
            nn - [num_classes, hidden_layer_units]
        cm - do you want to return matrix of accuracy
Returns:
"""

def mfold_cross_validation (group_indices, data, classifier_name,params=[],verbose=1,cm_flag=0):
    # TODO: make groups an optional parameter
    np.random.seed(0)

    total_correct = 0
    total_accuracy = 0
    acc_arr = []
    accuracies_matrix = np.zeros((group_indices.shape[0]))
    for i,g in enumerate(group_indices):
        mask = np.ones(data.shape[0],dtype=bool)
        mask[g] = 0
        tr = data[mask]
        te = data[g]
        #tr,te = pp.normalize(tr,te) # normalize the data


        if classifier_name == "case1":
            acc,cm,predicted = cls.discriminant_accuracy(tr,te,params[0],1)
        elif classifier_name == "case2":
            acc,cm,predicted = cls.discriminant_accuracy(tr,te,params[0],2)
        elif classifier_name == "case3":
            acc,cm,predicted = cls.discriminant_accuracy(tr,te,params[0],3)

        elif classifier_name == "knn":
            acc,cm,predicted = cls.knn_accuracy(tr,te,params[0],params[1],params[2],params[3]) # default to euclidean dist
        elif classifier_name == "knn_thread":
            acc,cm,predicted = cls.knn_threads(tr,te,params[0],params[1],params[2]) # default to euclidean dist

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
        accuracies_matrix[i] = acc

        acc_arr = np.append(acc_arr,acc)

        if verbose == 1:
            print("m",i,"correct",int(acc*te.shape[0]),"out of",te.shape[0])

#    return total_correct/data.shape[0]
    if cm_flag==0:
        return total_accuracy/group_indices.shape[0], np.std(acc_arr)
    else:
        return total_accuracy/group_indices.shape[0], np.std(acc_arr), accuracies_matrix


def naive_bayesian (tr, te, cm_array):
    label_array = []
    num_classifiers = len(cm_array)
    dims = int(np.max(tr[:,-1])+1)
    for cm in cm_array:
        cm = cm.astype("float32")/cm.sum(axis=1)[:,None]
        label_array.append(cm)
    indexes = list(itertools.product(range(dims),repeat=num_classifiers))

    lookup_table = np.zeros((dims,pow(dims,num_classifiers)))
    for i,comb in enumerate(indexes):
        product = label_array[0][:,comb[0]] # initalize the product
        for j in range(1,num_classifiers):
            product = product * label_array[j][:,comb[j]]
        lookup_table[:,i] = product.T

    np.set_printoptions(suppress=True)
    return lookup_table


def fusion (tr, te, cm_array, predicted_array):
    correct = 0
    lookup_table = naive_bayesian(tr,te,cm_array)
    num_classifiers = len(cm_array)
    dims = int(np.max(tr[:,-1])+1)
    indexes = list(itertools.product(range(dims),repeat=num_classifiers))
    confusion_matrix = np.zeros((dims,dims))
    predicted = np.zeros(te.shape[0])

    for test_index, test_sample in enumerate(te):
        print("Sample",test_index)
        votes = np.zeros(num_classifiers)
        for i, predicted in enumerate(predicted_array):
            votes[i] = predicted_array[i][test_index]
        column_index = indexes.index(tuple(votes))
        final_vote = np.argmax(lookup_table[:,column_index])

        confusion_matrix[int(test_sample[-1]),final_vote] += 1
        predicted[test_index] = final_vote
        if final_vote == test_sample[-1]:
            correct += 1

    return correct/te.shape[0], confusion_matrix, predicted

