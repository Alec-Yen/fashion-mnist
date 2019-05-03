import numpy as np
from scipy import stats
import random

from cedwar45.metrics import dist_from_clusters

def euc_dist(x,y):
#    return np.sqrt([a + b for a, b in zip(x,y)]);
    return np.linalg.norm(y-x, axis = 1);



def kMeans(X, k, d, seed = None):

    np.random.seed(seed)
    
    X_unique, unique_index = np.unique(X, return_index=True, axis=0)
    r_index = np.random.choice(unique_index, k, replace = False);
    clusters = X[r_index].astype(float);
    
    #clusters = X[cluster_index].copy();
    
    dist = np.zeros((k, X.shape[0]));
    
    #print(euc_dist(np.matrix([0,0]), np.matrix([1,1])))
    
    winners = np.zeros((X.shape[0]));
    
    MD = []
    
    z = 0;
    while True:
        print("Iter: " + str(z));
        z+=1;

        for j, x in enumerate(X):
            dist[:,j] = euc_dist(clusters,x);
        
        old_winners = winners.copy();
        winners = np.argmin(dist, axis = 0);
        
        #print((winners.shape == old_winners).all())
        
        print("Changed: " + str(X.shape[0] - sum(old_winners == winners)))
        if sum(old_winners == winners) == X.shape[0]:
            print('Done');
            break;
        
        c_means = np.zeros((k,d));
        
        for c in range(k):
            #print(np.mean(X[winners==c], axis = 0))#.shape)
            #if len(X[winners==c]): #X[winners==c] is not empty
            c_means[c] = np.mean(X[winners==c], axis = 0)
        
        clusters = c_means;
        
        print(dist_from_clusters(X, winners, clusters));
        MD.append(dist_from_clusters(X, winners, clusters))
    
        print();
    
    Y = np.zeros(X.shape);
    
    for c in range(k):
        Y[winners==c] = clusters[c];
    
    return Y, winners, MD
    
    
    
    
    
    
    