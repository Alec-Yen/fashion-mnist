import numpy as np

def ConfusionMatrix(predicted, gt, c): #gt is ground truth, c is class number
    CM = np.zeros([c,c])
    
    for i in range(len(predicted)):
        CM[gt[i], predicted[i]] += 1;
    
    return CM