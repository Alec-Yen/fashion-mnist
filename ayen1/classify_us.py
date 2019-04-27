import numpy as np
import ayen1.util as util
import itertools
import time


"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition
classify_us.py

Purpose: Functions
This file includes function definitions regarding unsupervised classification.
"""

############################################################


"""
Purpose: Evaluate performance of image compression using mean-squared error
Args:   x - original image
        y - compressed image
Returns: Calculates the error for each R, G, and B. Calculates the square. Returns the mean of all of these altogether.
"""
def mean_squared_error (x,y):
    return np.mean((x-y)**2)

"""
Purpose: Evaluate performance of image compression per R,G,B values
Args:   x - original image
        y - compressed image
Returns: Returns the mean-squared error of R,G, and B separately
"""
def rgb_mean_squared_err (x,y):
    r = np.mean((x[:,0] - y[:,0])**2) # can't use util.dist for this
    g = np.mean((x[:,1] - y[:,1])**2)
    b = np.mean((x[:,2] - y[:,2])**2)
    return r,g,b

"""
Purpose: Evaluate performance of image compression using visual greyscale
Args:   x - original image
        y - compressed image
Returns: Returns the distance (error) between two images per pixel as greyscale and inverts so that the least error is white and the most error is black
"""
def greyscale_error (x, y, max=-1):
    ret = util.dist(x, y)
    if max != -1:
        ret = max - ret
    else:
        ret = ret.max() - ret
    return ret



"""
Purpose: Find new mean cluster centers, based on the data samples assigned to it
Args:   labels - is same length as test_data, where each element indicates the index (label) of the centroid it goes with in the centroids array
        k - number of cluster centers
        test_data - the test data
Returns: centroids - of length k, each element has same dimension as elements in test_data
"""
def compute_centroids (test_data, k, labels):
    centroids = np.zeros((k,test_data.shape[1]))
    for i in range(0,k):
        temp = np.where((labels == i))
        label_data = test_data[temp]
        centroids[i] = np.mean(label_data,axis=0) # find mean of the data points
    return centroids

"""
Purpose: Find which centroid is the closest to a data sample x
Args:   x - a single data sample
        centroids - the cluster centers
Returns: the index of centroids that is nearest x
"""
def nearest_cluster (x, centroids):
    distances = util.dist(x,centroids)
    return np.argmin(distances)


"""
Purpose: Assign each test sample to nearest cluster
Args:   labels - is same length as test_data, where each element indicates the label of the centroid it goes with
        centroids - of length k, indicates the mean cluster centers
        test_data - the test data
Returns:    labels - new reassigned labels
            labels_difference - how many of the labels changed
"""
def assign_to_nearest_cluster (test_data, centroids, labels):
    original_labels = labels.copy()
    for i,x in enumerate(test_data):
        labels[i] = nearest_cluster (x,centroids)
    #return labels, util.dist(labels,original_labels,0)
    return labels, labels.size - np.sum(original_labels == labels)


"""
Purpose: Runs the k-means algortihm on a set of test data given k clusters and returns the clustered data
Args:   test_data - the test data
        k - number of clusters
        verbose - print out the epochs
        seed - seed random selection of initial clusters
Returns: the clustered data
"""
def k_means (test_data, k, verbose=False, seed=-1):
    start_time = time.time()
    ret = np.zeros(test_data.shape)
    if seed != -1:
        np.random.seed(seed)

    # must choose centroids that are unique
    test_data_unique, unique_index = np.unique(test_data,return_index=True,axis=0)
    random_index = np.random.choice(unique_index,k,replace=False) # choose k random indexes from dataset
    centroids = test_data[random_index].astype(float) # set those indexes as initial centroids

    # initialize variables
    labels = np.zeros(test_data.shape[0])
    labels_difference = 1

    # loop until the means don't change any more
    it = 1
    while labels_difference != 0:
        # reassign test samples to their nearest centroids
        labels, labels_difference = assign_to_nearest_cluster(test_data, centroids, labels)
        # after each epoch, recalculate centroids
        centroids = compute_centroids(test_data,k,labels) # after each epoch, recalculate centroids

        # print out epochs
        if verbose:
            print("Epoch: ",it," Label Change: ",labels_difference)
        it += 1

    # once the labels don't change anymore, each feature become their centroids
    for i,x in enumerate(test_data):
        ret[i] = centroids[labels[i].astype(int)]

    return ret, time.time()-start_time, it


"""
Purpose: Runs the winner-takes-all algorithm on a set of test data given k clusters and returns the clustered data
Args:   test_data - the test data
        k - number of clusters
        epsilon - resembles the learning rate, how quickly the algorithm converges
        verbose - print out the epochs
        seed - seed random selection of initial clusters
Returns: the clustered data
"""
def winner_takes_all (test_data, k, epsilon, verbose=False, seed=-1):
    start_time = time.time()
    ret = np.zeros(test_data.shape)
    if seed != -1:
        np.random.seed(seed)

    # must choose centroids that are unique
    test_data_unique, unique_index = np.unique(test_data,return_index=True,axis=0)
    random_index = np.random.choice(unique_index,k,replace=False) # choose random indexes from dataset
    centroids = test_data[random_index].astype(float) # set those indexes as initial centroids

    # initialize variables
    labels = np.zeros(test_data.shape[0])
    labels_difference = 1
    it = 1

    # loop until the means don't change
    while labels_difference != 0:
        # for each test sample x, move its nearest cluster towards it
        for i,x in enumerate(test_data):
            nearest_cluster_index = nearest_cluster(x, centroids)
            centroids[nearest_cluster_index] = centroids[nearest_cluster_index] + epsilon * (x - centroids[nearest_cluster_index])
        # reassign test samples to their nearest centroids
        labels, labels_difference = assign_to_nearest_cluster(test_data, centroids, labels)
        # after each epoch, recalculate centroids
        centroids = compute_centroids(test_data,k,labels)

        # print out epochs
        if verbose:
            print("Epoch: ",it," Label Change: ",labels_difference)
        it += 1

    # each feature become their centroids
    for i,x in enumerate(test_data):
        distances = util.dist(x,centroids)
        ret[i] = centroids[np.argmin(distances)]

    return ret, time.time()-start_time, it

"""
Purpose: Runs the Kohonen maps algorithm on a set of test data given k clusters and returns the clustered data
Args:   test_data - the test data
        k - number of clusters, must be a square for topology based
        epsilon - resembles the learning rate, how quickly the algorithm converges
        sigma - indicates how much you want to pull the cluster center's neighbors 
        flag - choose either "content" for content-based or "topology" for topology based
        max_time = in seconds, how long the algorithm is permitted to run, if -1 there is no limit
        max_epoch = max number of epochs allowed
        verbose - print out the epochs
        seed - seed random selection of initial clusters
Returns: the clustered data
"""
def kohonen_maps (test_data, k, epsilon, sigma, flag, max_epoch=-1, max_time=-1, verbose=False, seed=-1):
    start_time = time.time()
    ret = np.zeros(test_data.shape)
    if seed != -1:
        np.random.seed(seed)

    # must choose centroids that are unique
    test_data_unique, unique_index = np.unique(test_data,return_index=True,axis=0)
    random_index = np.random.choice(unique_index,k,replace=False) # choose random indexes from dataset
    centroids = test_data[random_index].astype(float) # set those indexes as initial centroids

    # error checking
    if not(np.sqrt(k).is_integer()) and flag == "topology":
        print("k must be a square of a number if using topology-based")
        return 1
    if flag != "topology" and flag != "content":
        print("invalid flag, must be either 'topology' or 'content'")
        return 1

    # set up a topology
    if flag == "topology":
        x = np.arange(np.sqrt(k))
        y = np.arange(np.sqrt(k))
        topology = np.array(list(itertools.product(x,y)))

    # initialize variables
    labels = np.zeros(test_data.shape[0])
    labels_difference = 1
    it = 1

    # loop until the means don't change
    while labels_difference != 0:
        # for each test sample x, move its nearest cluster towards it
        for i,x in enumerate(test_data):
            nearest_cluster_index = nearest_cluster(x, centroids)
            for j,c in enumerate(centroids): # pull all the centroids neighbors as well
                if flag == "topology":
                    phi = np.exp(-1*util.dist(topology[j],topology[nearest_cluster_index],ax=0)/sigma**2)
                elif flag == "content":
                    phi = np.exp(-1*util.dist(centroids[j],centroids[nearest_cluster_index],ax=0)/sigma**2)
                centroids[j] = centroids[j] + epsilon * phi * (x - centroids[nearest_cluster_index])
        # reassign test samples to their nearest centroids
        labels, labels_difference = assign_to_nearest_cluster(test_data, centroids, labels)
        # after each epoch, recalculate centroids
        centroids = compute_centroids(test_data,k,labels)

        # print out epochs
        if verbose:
            print("Epoch: ",it," Label Change: ",labels_difference)
        it += 1

        # break if exceeds max time
        if max_time != -1 and max_epoch != -1:
            if (time.time()-start_time) > max_time and it >= max_epoch:
                break
        if max_epoch != -1:
            if it >= max_epoch:
                break
        if max_time != -1:
            if (time.time()-start_time) > max_time:
                break

    # each feature become their centroids
    for i,x in enumerate(test_data):
        distances = util.dist(x,centroids)
        ret[i] = centroids[np.argmin(distances)]

    return ret, time.time()-start_time, it
