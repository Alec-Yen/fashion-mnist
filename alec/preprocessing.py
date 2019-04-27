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
    index = 0
    for i in tr_arr:  # for each class
        mu.append(np.mean(i,axis=0).reshape(-1,1))  # take average by axis
        cov_tmp = np.zeros((tr_arr[index].shape[1], tr_arr[index].shape[1]))
        for x in i:
            x = x.reshape((x.shape[0],1))
            cov_tmp += (x-mu[index]).dot((x-mu[index]).T)
        cov.append(np.true_divide(cov_tmp, tr_arr[index].shape[0]))
        index += 1
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
    cov_tmp = np.zeros((tr.shape[1], tr.shape[1]))
    for x in tr:
        x = x.reshape((x.shape[0],1))
        cov_tmp += (x-mu).dot((x-mu).T)
    cov = np.true_divide(cov_tmp, tr.shape[0])
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

    mu,cov = get_params(tr)

    ntrf = trf
    for x_index, x in enumerate(ntrf):
        for f_index,feature in enumerate(x):
            ntrf[x_index,f_index] = (feature - mu[f_index])/np.sqrt(cov[f_index,f_index])
    ntef = tef
    for x_index, x in enumerate(ntef):
        for f_index,feature in enumerate(x):
            ntef[x_index,f_index] = (feature - mu[f_index])/np.sqrt(cov[f_index,f_index])
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
            error = 0
            for i in range(m,eigenValues.size):
                error += eigenValues[i]/sum(eigenValues)
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

    tr_arr = return_2class_as_array(tr) # split up data into two classes
    mu = []
    scatter = []
    index = 0
    for i in tr_arr:  # for each class
        mu.append(np.mean(i,axis=0).reshape(-1,1))  # take average by axis
        scatter_tmp = np.zeros((tr_arr[index].shape[1], tr_arr[index].shape[1]))
        for x in i:
            x = x.reshape((x.shape[0],1))
            scatter_tmp += (x-mu[index]).dot((x-mu[index]).T)
        scatter.append(scatter_tmp)
        index += 1
    mu = np.asarray(mu)
    sw = sum(np.asarray(scatter)) # scatter matrix
    w = np.linalg.inv(sw).dot(mu[0]-mu[1]) # projection vector

    ftrf = trf.dot(w)
    ftef = tef.dot(w)
    ftr = append_column(ftrf,trl)
    fte = append_column(ftef,tel)
    return ftr,fte

