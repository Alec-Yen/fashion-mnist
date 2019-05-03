import numpy as np
import random

#from metrics import dist_from_clusters
from cedwar45.metrics import dist_from_clusters


def euc_dist(x,y):
    return np.linalg.norm(x - y, axis = 1);

def closest(x, clusters):
    return np.argmin(euc_dist(x,clusters));

def new_winners(X, clusters, winners):
    winners = np.zeros((winners.shape));
    for j, x in enumerate(X):
        winners[j] = closest(x,clusters);

    return winners


def WTA(X, k, epsilon, stop, seed = None):
    
    np.random.seed(seed)
    
    X_unique, unique_index = np.unique(X, return_index=True, axis=0)
    r_index = np.random.choice(unique_index, k, replace = False);
    clusters = X[r_index].astype(float);
    
    
    d = X.shape[1];
    
    #clusters = X[cluster_index];
    old_clusters = np.empty_like (clusters)
    
    dist = np.zeros((k, X.shape[0]));
    
    
    winners = np.zeros((X.shape[0]));
    
    MD = []
    
    z = 0;
    for i in range(stop):
        print("It: "+str(z));
        z+=1;
        
        
        for i,x in enumerate(X):
            #dist_ = np.zeros((clusters.shape[0]));
            
            #dist_ = euc_dist(x,clusters);
            
            #j = np.argmin(dist_);
            j = closest(x, clusters)
            clusters[j] = clusters[j] + epsilon * (x - clusters[j]);
            
        
        
        old_winners = winners.copy();
        winners = new_winners(X, clusters, winners);
        
        #print(dist_from_clusters(X, winners, clusters));
        if (old_winners == winners).all(): break;
        print("Changed: " + str(X.shape[0] - sum(old_winners == winners)))
        
        
        c_means = np.zeros((k,d));
        
        for c in range(k):
            #print(np.mean(X[winners==c], axis = 0))#.shape)
            #print(len(X[winners==c]))
            if len(X[winners==c]): #X[winners==c] is not empty
                data = X[np.where((winners == c))];   
                c_means[c] = np.mean(data, axis = 0);
        
        clusters = c_means;
        
        print(dist_from_clusters(X, winners, clusters));
        MD.append(dist_from_clusters(X, winners, clusters))
    
    Y = np.zeros(X.shape);
    
    for c in range(k):
        Y[winners==c] = clusters[c];
    
    return Y, winners, MD
    
    
    
    
    
    
    