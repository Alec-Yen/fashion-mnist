import numpy as np
from PIL import Image


def dist_sum_mean(img1, img2):
    return np.mean(np.mean(np.abs(img1 - img2)));

#mean squared error
def mean_squared_distance(img1, img2):
    return np.mean(np.mean((img1 - img2)**2));
    

#mean squared error rgb
def mean_squared_distance_rgb(img1, img2):
    r = np.mean(np.mean((img1[:,0] - img2[:,0])**2));
    g = np.mean(np.mean((img1[:,1] - img2[:,1])**2));
    b = np.mean(np.mean((img1[:,2] - img2[:,2])**2));
    
    return r,g,b

def greyscale_MSE(img1, img2):
    a = Image.fromarray(img1).convert('LA')
    b = Image.fromarray(img2).convert('LA')
    a = (np.array(a))[:,:,0];
    b = (np.array(b))[:,:,0];
    return mean_squared_distance(a,b)
    

def dist_from_clusters(X, labels, clusters):
    labels = labels.astype(int);
    sum = 0;

    for i,x in enumerate(X):
        #print(clusters[labels[i]].shape)
        sum += np.linalg.norm(x-clusters[labels[i]])
        
    sum /= X.shape[0];
    
    return sum;
