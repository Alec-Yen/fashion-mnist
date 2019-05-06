import numpy as np
import matplotlib.pyplot as plt

"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition
Final Project
main_ayen1.py

Purpose: Main Program
Description
"""

# files with my function definitions
import ayen1.perfeval as pe
import ayen1.util as util
import ayen1.classify_s as cls
import ayen1.classify_us as clu
import ayen1.preprocessing as pp

NUM_CLASSES = 10
prior_arr = np.repeat(1.0/NUM_CLASSES,NUM_CLASSES)

def case_tests (tr,te):
    case1acc,case1cm,case1predicted = cls.discriminant_accuracy(tr,te,prior_arr,1,verbose=0)
    print("Case 1 Acc",case1acc)
    print(case1cm)
    case2acc,case2cm,case2predicted = cls.discriminant_accuracy(tr,te,prior_arr,2,verbose=0)
    print("Case 2 Acc",case2acc)
    print(case2cm)
    case3acc,case3cm,case3predicted = cls.discriminant_accuracy(tr,te,prior_arr,3,verbose=0)
    print("Case 3 Acc",case3acc)
    print(case3cm)

    return case1acc, case1cm, case1predicted, case2acc, case2cm, case2predicted, case3acc, case3cm, case3predicted





##############LOAD DATA#################################

import zalando.utils.mnist_reader as mnist_reader
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


y_train = y_train.reshape(y_train.size,1)
y_test = y_test.reshape(y_test.size,1)

tr = np.hstack((X_train.astype("float32")/255,y_train))
te = np.hstack((X_test.astype("float32")/255,y_test))



#######################PREPROCESSING##################################

ptr, pte, perr = pp.pca(tr,te,0.1,0)
ftr, fte = pp.fld(tr,te)


############################ CLASSIFIER TESTING ############################################

k = 5 # for KNN
p = 3 # the order of the Minkowskit distance
num_threads = 10
k_arr = [5] # the next one
#k_arr = [10, 20, 50, 100, 250]

print("Raw")
# case1acc, case1cm, case1predicted, case2acc, case2cm, case2predicted, case3acc, case3cm, case3predicted = case_tests(tr,te)
# util.writeToFile('data/case1_acc_raw.txt',case1acc,4)
# util.writeToFile('data/case2_acc_raw.txt',case2acc,4)
# util.writeToFile('data/case3_acc_raw.txt',case3acc,4)
# np.savetxt('data/case1_cm_raw.txt',case1cm,fmt='%d')
# np.savetxt('data/case2_cm_raw.txt',case2cm,fmt='%d')
# np.savetxt('data/case3_cm_raw.txt',case3cm,fmt='%d')
# np.savetxt('data/case1_predicted_raw.txt',case1predicted,fmt='%d')
# np.savetxt('data/case2_predicted_raw.txt',case2predicted,fmt='%d')
# np.savetxt('data/case3_predicted_raw.txt',case3predicted,fmt='%d')

# knn_acc, knn_cm, knn_predicted = cls.knn_threads(tr,te,k,num_threads,p)
# util.writeToFile('data/knn_acc_raw_k'+str(k)+'_p'+str(p)+'.txt',knn_acc,4)
# np.savetxt('data/knn_cm_raw_k'+str(k)+'_p'+str(p)+'.txt',knn_cm,fmt='%d')
# np.savetxt('data/knn_predicted_raw_k'+str(k)+'_p'+str(p)+'.txt',knn_predicted,fmt='%d')

print("PCA")
# case1acc, case1cm, case1predicted, case2acc, case2cm, case2predicted, case3acc, case3cm, case3predicted = case_tests(ptr,pte)
# util.writeToFile('data/case1_acc_pca.txt',case1acc,4)
# util.writeToFile('data/case2_acc_pca.txt',case2acc,4)
# util.writeToFile('data/case3_acc_pca.txt',case3acc,4)
# np.savetxt('data/case1_cm_pca.txt',case1cm,fmt='%d')
# np.savetxt('data/case2_cm_pca.txt',case2cm,fmt='%d')
# np.savetxt('data/case3_cm_pca.txt',case3cm,fmt='%d')
# np.savetxt('data/case1_predicted_pca.txt',case1predicted,fmt='%d')
# np.savetxt('data/case2_predicted_pca.txt',case2predicted,fmt='%d')
# np.savetxt('data/case3_predicted_pca.txt',case3predicted,fmt='%d')

# knn_acc, knn_cm, knn_predicted = cls.knn_threads(ptr,pte,k,num_threads,p)
# util.writeToFile('data/knn_acc_pca_k'+str(k)+'_p'+str(p)+'.txt',knn_acc,4)
# np.savetxt('data/knn_cm_pca_k'+str(k)+'_p'+str(p)+'.txt',knn_cm,fmt='%d')
# np.savetxt('data/knn_predicted_pca_k'+str(k)+'_p'+str(p)+'.txt',knn_predicted,fmt='%d')

print("FLD")
# case1acc, case1cm, case1predicted, case2acc, case2cm, case2predicted, case3acc, case3cm, case3predicted = case_tests(ftr,fte)
# util.writeToFile('data/case1_acc_fld.txt',case1acc,4)
# util.writeToFile('data/case2_acc_fld.txt',case2acc,4)
# util.writeToFile('data/case3_acc_fld.txt',case3acc,4)
# np.savetxt('data/case1_cm_fld.txt',case1cm,fmt='%d')
# np.savetxt('data/case2_cm_fld.txt',case2cm,fmt='%d')
# np.savetxt('data/case3_cm_fld.txt',case3cm,fmt='%d')
# np.savetxt('data/case1_predicted_fld.txt',case1predicted,fmt='%d')
# np.savetxt('data/case2_predicted_fld.txt',case2predicted,fmt='%d')
# np.savetxt('data/case3_predicted_fld.txt',case3predicted,fmt='%d')
#

# knn_acc, knn_cm, knn_predicted = cls.knn_threads(ftr,fte,k,num_threads,p)
# util.writeToFile('data/knn_acc_fld_k'+str(k)+'_p'+str(p)+'.txt',knn_acc,4)
# np.savetxt('data/knn_cm_fld_k'+str(k)+'_p'+str(p)+'.txt',knn_cm,fmt='%d')
# np.savetxt('data/knn_predicted_fld_k'+str(k)+'_p'+str(p)+'.txt',knn_predicted,fmt='%d')



################################ KNN TESTING FOR DIFFERENT Ks ######################################


# knn_acc_arr = np.zeros(len(k_arr))
# print("KNN Raw")
# for i,k_tmp in enumerate(k_arr):
#     knn_acc_arr[i] = cls.knn_threads(tr, te, k_tmp, num_threads,2)[0]
#     util.writeToFile('data/knn_acc_raw_k'+str(k_tmp)+'_p'+str(p)+'.txt',knn_acc_arr[i],4)
# print(knn_acc_arr)
# plt.scatter(k_arr, knn_acc_arr)
# plt.ylabel("KNN Accuracy")
# plt.xlabel("k")
# plt.savefig("figures/knn/knn_acc_raw.png")
# plt.show()

# knn_acc_arr = np.zeros(len(k_arr))
# print("KNN PCA")
# for i,k_tmp in enumerate(k_arr):
#     knn_acc_arr[i] = cls.knn_threads(ptr, pte, k_tmp, num_threads,2)[0]
#     util.writeToFile('data/knn_acc_pca_k'+str(k_tmp)+'_p'+str(p)+'.txt',knn_acc_arr[i],4)
# print(knn_acc_arr)
# plt.scatter(k_arr, knn_acc_arr)
# plt.ylabel("KNN Accuracy")
# plt.xlabel("k")
# plt.savefig("figures/knn/knn_acc_pca.png")
# plt.show()

# knn_acc_arr = np.zeros(len(k_arr))
# print("KNN FLD")
# for i,k_tmp in enumerate(k_arr):
#     knn_acc_arr[i] = cls.knn_threads(ftr, fte, k_tmp, num_threads,2)[0]
#     util.writeToFile('data/knn_acc_fld_k'+str(k_tmp)+'_p'+str(p)+'.txt',knn_acc_arr[i],4)
# print(knn_acc_arr)
# plt.scatter(k_arr, knn_acc_arr)
# plt.ylabel("KNN Accuracy")
# plt.xlabel("k")
# plt.savefig("figures/knn/knn_acc_fld.png")
# plt.show()


######################### M-FOLD CROSS VALIDATION ###############################################

# split up into groups for 10-fold cross validation
np.random.seed(0)
indexes = np.arange(tr.shape[0])
np.random.shuffle(indexes)
m = 10
groups = []
per_group = int(indexes.shape[0]/m)
for i in range(0,m):
    groups.append(indexes[i*per_group:(i+1)*per_group])
groups = np.array(groups)

print("M-Fold Raw")
# acc,std  = pe.mfold_cross_validation(groups,tr,"case1",params=[prior_arr])
# util.writeToFile('data/case1_mfold_raw.txt',acc,4)
# print("Case 1",acc)
# acc,std  = pe.mfold_cross_validation(groups,tr,"case2",params=[prior_arr])
# util.writeToFile('data/case2_mfold_raw.txt',acc,4)
# print("Case 2",acc)
# acc,std  = pe.mfold_cross_validation(groups,tr,"case3",params=[prior_arr])
# util.writeToFile('data/case3_mfold_raw.txt',acc,4)
# print("Case 3",acc)

# knn_acc_arr = np.zeros(len(k_arr))
# for i,k_tmp in enumerate(k_arr):
#     knn_acc_arr[i],std  = pe.mfold_cross_validation(groups,tr,"knn_thread",params=[k_tmp,num_threads,p])
#     util.writeToFile('data/knn_mfold_raw_k'+str(k_tmp)+'_p'+str(p)+'.txt',knn_acc_arr[i],4)
# print("KNN",knn_acc_arr)


print("M-Fold PCA")
# acc,std  = pe.mfold_cross_validation(groups,ptr,"case1",params=[prior_arr])
# util.writeToFile('data/case1_mfold_pca.txt',acc,4)
# print("Case 1",acc)
# acc,std  = pe.mfold_cross_validation(groups,ptr,"case2",params=[prior_arr])
# util.writeToFile('data/case2_mfold_pca.txt',acc,4)
# print("Case 2",acc)
# acc,std  = pe.mfold_cross_validation(groups,ptr,"case3",params=[prior_arr])
# util.writeToFile('data/case3_mfold_pca.txt',acc,4)
# print("Case 3",acc)

# knn_acc_arr = np.zeros(len(k_arr))
# for i,k_tmp in enumerate(k_arr):
#     knn_acc_arr[i],std  = pe.mfold_cross_validation(groups,ptr,"knn",params=[k_tmp,0,0,p])
#     util.writeToFile('data/knn_mfold_pca_k'+str(k_tmp)+'_p'+str(p)+'.txt',knn_acc_arr[i],4)
# print("KNN",knn_acc_arr)



print("M-Fold FLD")
# acc,std  = pe.mfold_cross_validation(groups,ftr,"case1",params=[prior_arr])
# util.writeToFile('data/case1_mfold_fld.txt',acc,4)
# print("Case 1",acc)
# acc,std  = pe.mfold_cross_validation(groups,ftr,"case2",params=[prior_arr])
# util.writeToFile('data/case2_mfold_fld.txt',acc,4)
# print("Case 2",acc)
# acc,std  = pe.mfold_cross_validation(groups,ftr,"case3",params=[prior_arr])
# util.writeToFile('data/case3_mfold_fld.txt',acc,4)
# print("Case 3",acc)
# acc,std  = pe.mfold_cross_validation(groups,ftr,"knn",params=[5,0,0,2])
# util.writeToFile('data/knn_mfold_fld_k'+str(k)+'_p'+str(p)+'.txt',acc,4)
# print("KNN",acc)

# knn_acc_arr = np.zeros(len(k_arr))
# for i,k_tmp in enumerate(k_arr):
#     knn_acc_arr[i],std  = pe.mfold_cross_validation(groups,ftr,"knn",params=[k_tmp,0,0,p])
#     util.writeToFile('data/knn_mfold_fld_k'+str(k_tmp)+'_p'+str(p)+'.txt',knn_acc_arr[i],4)
# print("KNN",knn_acc_arr)


############################# CLASSIFIER FUSION ##############################################

# fusion-1: KNN (PCA), BPNN (8 Hidden Nodes, Raw), CNN
# fusion-2: Case 2 (PCA), KNN (k=5), Decision Trees (Raw)



# # fusion-1
# cm_array = []
# predicted_array = []
# cm_array.append(np.loadtxt('data/knn_cm_pca_k'+str(k)+'_p'+str(p)+'.txt',dtype=int))
# cm_array.append(np.loadtxt('data/k3NN_h8_cm_raw.txt',dtype=int))
# cm_array.append(np.loadtxt('data/CNN_cm_raw.txt',dtype=int))
# predicted_array.append(np.loadtxt('data/knn_predicted_pca_k'+str(k)+'_p'+str(p)+'.txt',dtype=int))
# predicted_array.append(np.loadtxt('data/k3NN_h8_predicted_raw.txt',dtype=int))
# predicted_array.append(np.loadtxt('data/CNN_predicted_raw.txt',dtype=int))
# acc, cm, predicted = pe.fusion(tr, te, cm_array, predicted_array)
#
# util.writeToFile('data/fusion_acc_1.txt',acc,4)
# np.savetxt('data/fusion_cm_1.txt',cm,fmt='%d')
# np.savetxt('data/fusion_predicted_1.txt',predicted,fmt='%d')


# # fusion-2
# cm_array = []
# predicted_array = []
# cm_array.append(np.loadtxt('data/case2_cm_pca.txt',dtype=int))
# cm_array.append(np.loadtxt('data/knn_cm_pca_k5_p2.txt',dtype=int))
# cm_array.append(np.loadtxt('data/DTree_cm_raw.txt',dtype=int))
# predicted_array.append(np.loadtxt('data/case2_predicted_pca.txt',dtype=int))
# predicted_array.append(np.loadtxt('data/knn_predicted_pca_k5_p2.txt',dtype=int))
# predicted_array.append(np.loadtxt('data/DTree_predicted_raw.txt',dtype=int))
# acc, cm, predicted = pe.fusion(tr, te, cm_array, predicted_array)
#
# util.writeToFile('data/fusion_acc_2.txt',acc,4)
# np.savetxt('data/fusion_cm_2.txt',cm,fmt='%d')
# np.savetxt('data/fusion_predicted_2.txt',predicted,fmt='%d')


############################ MISCLASSIFIED ########################################

dictionary = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

# predicted = np.loadtxt('data/fusion_predicted_1.txt',dtype=int)
# max_to_show = 10
# count = 0
# for i,sample in enumerate(predicted):
#     if sample != te[i,-1]:
#         img = X_test[i].reshape((28,28))
#         plt.imshow(img,cmap='gray')
#         plt.title("Predicted Label: "+dictionary[sample]+"; True Label: "+dictionary[te[i,-1]])
#         plt.savefig("figures/misclassified/im"+str(count)+".png")
#         plt.show()
#         count += 1
#         if count == max_to_show:
#             break


epsilon = 0.01
test = fte.copy()
cluster_dict = {}
labels = clu.k_means(test,NUM_CLASSES,verbose=True,seed=0)[0]
#labels = clu.winner_takes_all(test,NUM_CLASSES,epsilon,verbose=True,seed=0)[0]
clusters = []
for i in range(NUM_CLASSES):
    clusters.append(np.where(labels==i)[0])

for i,c in enumerate(clusters):
    counts = np.bincount(test[c,-1].astype(int))
    cluster_dict[i] = np.argmax(counts)

print(cluster_dict)

correct = 0
for i,test_sample in enumerate(test):
    if cluster_dict[labels[i]] == test_sample[-1]:
        correct += 1
print(correct/test.shape[0])
