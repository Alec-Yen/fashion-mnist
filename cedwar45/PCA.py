import numpy as np
from scipy import stats

def PCA(tr_features, te_features, error):
    
    #number of dimensions
    d = tr_features.shape[1];
    
    #sizes
    sz_tr = tr_features.shape[0];
    sz_te = te_features.shape[0];
    
    #cov matrix
    Sigma = np.cov(tr_features.T);
    
    
    S, U = np.linalg.eig(Sigma);
    indices = np.argsort(S)[::-1];
    
    S = S[indices];
    
    U = U[:,indices];
    
    #Calculate the error to pick K, the number of eigenvectors we're keeping
    esum = 0;
    for i in range(d):
        esum = esum + S[d-i-1]/np.sum(S);
        if esum < error:
            K = d-i-1;
            
    tr_proj = np.zeros((sz_tr,K));
    te_proj = np.zeros((sz_te,K));
    
    for i,tr in enumerate(tr_features):
        for k in range(K):
            tr = np.matrix(tr);
            U_mat = np.matrix(U[:,k]).T;
            
            projection_k = tr * U_mat;
            
            tr_proj[i,k] = projection_k;
        
    #print(tr_proj.shape);
    
    for i,te in enumerate(te_features):
        for k in range(K):
            te = np.matrix(te);
            U_mat = np.matrix(U[:,k]).T;
            
            projection_k = te * U_mat;
            
            te_proj[i,k] = projection_k;
            
    #print(te_proj.shape);
    
    return tr_proj, te_proj;
    
    
#K is the number of eigenvectors to project onto
def PCA_k(tr_features, te_features, K):
    
    d = tr_features.shape[1];
    
    sz_tr = tr_features.shape[0];
    sz_te = te_features.shape[0];
    
    
    Sigma = np.cov(tr_features.T);
    
    
    S, U = np.linalg.eig(Sigma);
    indices = np.argsort(S)[::-1];
    S = S[indices];
    
    U = U[:,indices];
    
    
    tr_proj = np.zeros((sz_tr,K));
    te_proj = np.zeros((sz_te,K));
    
    for i,tr in enumerate(tr_features):
        for k in range(K):
            tr = np.matrix(tr);
            U_mat = np.matrix(U[:,k]).T;
            #print(tr.shape, U_mat.shape);
            projection_k = tr * U_mat;
            #print(projection_k);
            tr_proj[i,k] = projection_k;
        
    #print(tr_proj.shape);
    
    for i,te in enumerate(te_features):
        for k in range(K):
            te = np.matrix(te);
            U_mat = np.matrix(U[:,k]).T;
            
            projection_k = te * U_mat;
            
            te_proj[i,k] = projection_k;
            
    #print(te_proj.shape);
    
    return tr_proj, te_proj;