from sklearn import tree
import numpy as np

#M-fold
def skLearn_tree_class(tr_features, te_features, tr_classes):
    np.random.seed(321)
    clf = tree.DecisionTreeClassifier();
    
    clf.fit(tr_features, tr_classes)  
    predicted = clf.predict(te_features);
    return predicted;