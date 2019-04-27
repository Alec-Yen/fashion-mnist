import numpy as np
"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition
preprocessing.py

Purpose: Functions
This file includes function definitions regarding preprocessing, including
maximum likelihood estimation (parameter estimation), normalization, and dimensionality
reduction (PCA and FLD).
"""

############################################################

"""
Purpose:
    Separates data into different classes 
    Assumes binary classification (0 and 1)
Args:
    data: data to be separated
Returns:
    array of data, with number of elements equal to number of classes
"""
def return_2class_as_array(data):
    temp = np.where((data[:, -1] == 0))
    data1 = data[temp, 0:-1]
    temp = np.where((data[:, -1] == 1))
    data2 = data[temp, 0:-1]
    data1 = np.reshape(data1, (data1.shape[1], data1.shape[2]))
    data2 = np.reshape(data2, (data2.shape[1], data2.shape[2]))
    ret = np.array((data1, data2))
    return ret

"""
Purpose:
    Separates data into different classes 
Args:
    data: data to be separated
    num_classes: number of classes
Returns:
    array of data, with number of elements equal to number of classes
"""
def return_multiclass_as_array(data,num_classes):
    ret = []
    for i in range(num_classes):
        temp = np.where((data[:,-1] == i))
        data_n = data[temp,0:-1] # ignore class label
        data_n = np.reshape(data_n, (data_n.shape[1], data_n.shape[2]))
        ret.append(data_n)
    return ret


"""
Purpose:
    Append vector as column to ndarray
    Useful for appending class label column to feature data
Args:
    matrix: array to be appended to
    column: array to be appended
Returns:
    combined array
"""
def append_column(matrix,column):
    return np.hstack((matrix, column.reshape((column.shape[0], 1))))


"""
Purpose:
    Get Gaussian parameters of array of data, already separated into classes
    Works for any number of classes
Args:
    tr_arr: array of data, without class label column
Returns:
    mu: array of mean vectors
    cov: array of covariance matrices
"""
def get_params_arr(tr_arr):
    mu = list()
    cov = list()
    for i in tr_arr:
        mu.append(np.mean(i,axis=0).reshape(-1,1))  # take average by axis
        cov.append(np.cov(i.T,bias=False))
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    return mu,cov


"""
Purpose:
    Get Gaussian parameters of all data
Args:
    tr: array of data, without class label column (probably training data)
Returns:
    mu: mean vector
    cov: covariance matrix
"""
def get_params(tr):
    mu = np.mean(tr, axis=0).reshape(-1, 1)  # take average by axis
    cov = np.cov(tr.T,bias=False)
    return mu,cov


"""
Purpose:
    Normalize data to be (x_i-mu_i)/stddev
    Unsupervised, no knowledge of class labels
Args:
    tr: training data
    te: testing data
Returns:
    training data, normalized with tr params
    testing data, normalized with tr params
"""
def normalize(tr, te):
    trf = tr[:,:-1]
    tef = te[:,:-1]
    trl = tr[:,-1]
    tel = te[:,-1]

    ntrf = trf.copy() # need the copy or else it changes tr, te by reference
    ntef = tef.copy()
    for col in range(trf.shape[1]):
        ntrf[:,col] = (ntrf[:,col]-np.mean(ntrf[:,col]))/np.std(ntrf[:,col])
        ntef[:,col] = (ntef[:,col]-np.mean(ntef[:,col]))/np.std(ntef[:,col])

    # mu,cov = get_params(trf)
    # ntrf = trf
    # for x_index, x in enumerate(ntrf):
    #     for f_index,feature in enumerate(x):
    #         ntrf[x_index,f_index] = (feature - mu[f_index])/np.sqrt(cov[f_index,f_index])
    # ntef = tef
    # for x_index, x in enumerate(ntef):
    #     for f_index,feature in enumerate(x):
    #         ntef[x_index,f_index] = (feature - mu[f_index])/np.sqrt(cov[f_index,f_index])

    ntr = append_column(ntrf,trl)
    nte = append_column(ntef,tel)
    return ntr,nte

"""
Purpose:
    Implement principal component analysis (PCA)
    Unsupervised, no knowledge of class labels
Args:
    tr: training data, with class labels
    te: testing data, with class labels
    err_thr_or_m: either maximum error limit or number of dimensions to reduce to
    flag: 0 or 1
        flag=0: determine new dims m using max error limit
        flag=1: use the provided new dims m
Returns:
    training data, basis vectors applied
    testing data, basis vectors applied
"""
def pca(tr, te, err_thr_or_m, flag):
    trf = tr[:,:-1]
    tef = te[:,:-1]
    trl = tr[:,-1] # unused since unsupervised
    tel = te[:,-1]

    mu, cov = get_params(trf)
    eigenValues, eigenVectors = np.linalg.eig(cov)
    index = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[index] # sorted from greatest to smallest
    eigenVectors = eigenVectors[:,index]

    if flag == 0:
        err_exceeded = 0
        max_error = 0
        m = eigenValues.size-1
        while err_exceeded == 0:
            error = sum(eigenValues[m:]/sum(eigenValues))
            if error > err_thr_or_m:
                err_exceeded = 1
                m += 1 # return to the value that was still beneath the threshold
            else:
                max_error = error # save the error that was beneath the threshold
                m -= 1
    else:
        m = err_thr_or_m
        max_error = 0
        for i in range(m,eigenValues.size):
            max_error += eigenValues[i]/sum(eigenValues)
    P = eigenVectors[:,:m] # basis vector to apply

    ptrf = trf.dot(P) # nx7 by 7xm, apply to testing data
    ptef = tef.dot(P)
    ptr = append_column(ptrf,trl)
    pte = append_column(ptef,tel)
    return ptr, pte, max_error


"""
Purpose:
    Implement Fisher's linear discriminant (FLD)
    Supervised, has knowledge of class labels
Args:
    tr: training data, with class labels
    te: testing data, with class labels
Returns:
    training data, projected onto w
    testing data, projected onto w
"""
def fld(tr, te):
    trf = tr[:,:-1]
    tef = te[:,:-1]
    trl = tr[:,-1]
    tel = te[:,-1]

    num_classes = np.amax(te[:,-1])+1
    tr_arr = return_multiclass_as_array(tr,num_classes) # split up data into two classes
    mu_all, cov_all = get_params(trf) # give to get_params without features
    mu, cov = get_params_arr(tr_arr)
    scatter = cov * (te.shape[0]-1)

    mu = np.asarray(mu)
    sw = sum(np.asarray(scatter)) # scatter matrix
    sb = np.zeros(sw.shape)

    for k in range(num_classes):
        sb += tr_arr[k].shape[0] * (mu[k]-mu_all).dot((mu[k]-mu_all).T)
    if scatter.shape[0] <= 2:
        w = np.linalg.inv(sw).dot(mu[0]-mu[1]) # projection vector
    else: # more than 2D
        eigenValues, eigenVectors = np.linalg.eig(np.linalg.pinv(sw).dot(sb))
        index = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[index] # sorted from greatest to smallest
        eigenVectors = eigenVectors[:,index]
        w = eigenVectors[:,:num_classes-1]


    ftrf = trf.dot(w)
    ftef = tef.dot(w)
    ftr = append_column(ftrf,trl)
    fte = append_column(ftef,tel)
    return ftr,fte

