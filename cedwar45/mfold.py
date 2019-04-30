import numpy as np
from scipy import stats
import time

import itertools # see https://stackoverflow.com/questions/1720421/how-do-i-concatenate-two-lists-in-python

def accuracy(predicted, te_classes, c = 2):
    n_te = predicted.shape[0];
    #te = np.zeros((c,1));

    #te[0] = sum(te_classes==0);
    #te[1] = sum(te_classes==1);

    rv = (sum(predicted==te_classes)/n_te); #total accuracy
    #index = np.where(te_classes==0);
    #rv_0 = sum(predicted[index] == te_classes[index])/te[0]; #class 0 accuracy
    #index = np.where(te_classes==1);
    #rv_1 = sum(predicted[index] == te_classes[index])/te[1]; #class 1 accuracy

    return rv#, rv_0, rv_1;


def mfold(X_grp, X_classes, classifier):
    
    m = len(X_grp);
    
    acc_sum = 0;
    acc_sum_sk = 0;

    accs = []; #keep track of acc for each m
    
    X_train = []; #This is inefficient so it won't scale extremely. See itertools above maybe
    for i in range(m): 
        X_tr = np.array([]);
        X_tr_classes = np.array([]);
        for j in range(m):
            if i == j:
                X_te = X_grp[j];
                X_te_classes = X_classes[j];
                continue
            
            X_tr = np.vstack([X_tr, X_grp[j]]) if X_tr.size else X_grp[j];
            #print(X_tr.shape, X_grp[j].shape)
            X_tr_classes = np.append(X_tr_classes, X_classes[j]);
        
        
        
        #print(X_tr.shape, X_tr_classes.shape, X_te.shape)
        predicted = classifier(X_tr, X_te, X_tr_classes);
        acc = accuracy(predicted, X_te_classes);
        accs.append(acc);
        print(acc)
        acc_sum += acc;
        
     
    #Print for latex table
    #for i in accs:
    #    print("{:.1f}".format(i*100), "&", end=" ")
    #print()
    
    return acc_sum/m, np.std(accs);
    
    tstart = time.time();
    
    
    te_classes = np.zeros(te_features.shape[0]);
    
    for m,te_ in enumerate(te_features):
        te = np.matrix(te_);
        d = [];
        for tr_ in tr_features:
            tr = np.matrix(tr_);
            d.append(np.linalg.norm(te-tr));
            
        indices = np.argsort(d)[0:k];
        classes = tr_classes[indices];
        
        
        num0 = sum(classes == 0);
        num1 = sum(classes == 1);
        
        
        cmp0 = num0/k;
        cmp1 = num1/k;
        
        if cmp0 == cmp1: te_classes[m] = np.random.randint(0,2);
        elif cmp0 < cmp1: te_classes[m] = 1;
        #else: te_classes[m] = 0; #already done
        
        
        #te_classes[m] = stats.mode(classes)[0];
        
    tend = time.time();
    
    total = tend-tstart; #total time taken
        
    return te_classes;
    