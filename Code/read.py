from __future__ import division
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature_exctraction import extract_features #  importerar function 
#from printHelper import print_and_save_content
import PCA
import glob2
import wfdb
import numpy as np
import scipy.io as sio
import h5py
#import csv
import time
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import multiprocessing 



import sklearn
dirList_2 = glob2.glob("/Users/jacobstachowicz/Downloads/TRAIN/tr*")


def read(PCA_v = True, covariances = None, begin=0, end=10):
  prev = None
  X    = None
  Y    = None

  for i in range(begin,end):
      sample  = dirList_2[i][-10:]
      mat_string = dirList_2[i] + sample + '.mat'
      arousal_string = dirList_2[i] + sample
      print("Working on sample nr: ",  i, )
      
      x = sio.loadmat(mat_string)['val']

      if (PCA_v):
        ann = wfdb.rdann(arousal_string, 'arousal')
        prev = PCA.get_matrices(x,ann,prev)
      
      else:
        arousal_string += '-arousal.mat'     
        f = h5py.File(arousal_string, 'r')
        y = f['data']['arousals'][:]
        X,Y = extract_features(x,np.transpose(y),covariances,X,Y)
  print("-----------------------")

  if(PCA_v):
    return prev
  return X,Y      



#-----------------------------------------------------------------#
#---------------------------- SETUP ------------------------------#
start = 1                                                         
stop = 14                                                        
steps = 2                                                      
array_size = len(range(start,stop,steps))                         
print("Array_sizzeuu: ", array_size)                              
#array_size = int(round((stop-start)/steps))                      
#amount = len(dirList_2)                                          
amount = 5                                                        
test_begin = 4                                                    
#-----------------------------------------------------------------#
#-----------------------------------------------------------------#




test_end = amount
train_begin = 0
train_end = test_begin


cov_train = PCA.PCA(read(end=train_end))
cov_test = PCA.PCA(read(begin=test_begin, end=test_end))

X_train,Y_train = read(PCA_v = False, covariances = cov_train, end = train_end)
X_test,Y_test = read(PCA_v = False, covariances = cov_test, begin=test_begin, end=test_end)
print("array size = ", array_size)
ann_aurpc = [0] *array_size
svm_aurpc = [0] *array_size
rfc_aurpc = [0] *array_size
knn_aurpc = [0] *array_size

knn_new = [0] *array_size
step_count = 0;


    
def multiprocessing_func(index, a, s, s4, r, k):
    
    starttime = time.time()
    
    num_f = index * steps
    from sklearn.feature_selection import SelectKBest
    #from sklearn.feature_selection import chi2
    from sklearn.feature_selection import f_classif # denna klarar av negativa värden

    #from sklearn.neural_network import MLPClassifier
    #classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    #test = optimize_select_features(X_train, Y_train, X_test, Y_test, classifier)
    feature_select = True
    if(feature_select):
      num_f = index * steps

      # Feature extraction
      extracted = SelectKBest(score_func=f_classif, k=num_f)
    
      #Nedan har jag lagt till för den klagar på negativa värden
      # ValueError: Input X must be non-negative. https://stackoverflow.com/questions/25792012/feature-selection-using-scikit-learn
      from sklearn import preprocessing
      normalized_X = preprocessing.normalize(X_train)
    
      fit = extracted.fit(normalized_X, Y_train)
    
      # Summarize scores
      #np.set_printoptions(precision=15)
      #print(fit.scores_)
      print("Total amount of features: ", len(fit.scores_ ))
    
      #features = fit.transform(X_train)
      # Summarize selected features
      #print(features[0:5,:])
    
      #  Recursive Feature Elimination, a type of wrapper feature selection method.
    
      from sklearn.feature_selection import RFE
      from sklearn.linear_model import LogisticRegression

      # Feature extraction
      import warnings
      warnings.filterwarnings("ignore", category=FutureWarning)
      model = LogisticRegression()
      # LogisticRegression(solver='lbfgs')
      rfe = RFE(model, num_f)

      #Class compensate
      idx1 = [i for i, x in enumerate(Y_train) if x == 1]
    
      idx0 = [i for i, x in enumerate(Y_train) if x == 0]
    
      x_comp = np.array(X_train)[idx1]
      y_comp = np.array(Y_train)[idx1]

      correcting_amount = 2
    
      x_comp1 = np.array(X_train)[idx0][:len(idx1)*correcting_amount]
      y_comp1 = np.array(Y_train)[idx0][:len(idx1)*correcting_amount]
    
      X_train_comp = np.append(x_comp, x_comp1, axis = 0)
      Y_train_comp = np.append(y_comp, y_comp1, axis = 0)
    
      #Scale data
      X_train_comp_sc = sc.fit_transform(X_train_comp)
      X_test_sc = sc.fit_transform(X_test)

      print('Start feature selection for the best ', num_f, " features" )
    
      fit = rfe.fit(X_train_comp_sc, Y_train_comp)

      ix = [i for i, x in enumerate(fit.support_) if not x]
      X_train_comp_e = np.delete(X_train_comp_sc, ix, 1)
      X_test_comp_e = np.delete(X_test_sc, ix,1)
    
      print('Finished feature selectionor the best ', num_f, " features" )


    ##############################################

    # Save the data to file
    #w = csv.writer(open("output.csv", "w"))
    #for key, val in t.items():
    #    w.writerow([key, val])

    # read the data to file
    #with open('output.csv') as f:
    #   mustard = dict(dict(list(filter(None, csv.reader(f)))))

    #print_and_save_content(t)

    #Declare classifiers
    from sklearn.svm import SVC
    SVM = SVC(C=3, max_iter=-1)    
    SVM4 = SVC(kernel='linear', C=2.0, random_state=0, max_iter=-1)

    from sklearn.neural_network import MLPClassifier

    ANN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

    from sklearn.ensemble import RandomForestClassifier

    RFC = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=3)

    #Train classifiers
    ANN.fit(X_train_comp_e,Y_train_comp)
    SVM.fit(X_train_comp_e, Y_train_comp)
    SVM4.fit(X_train_comp_e, Y_train_comp)
    RFC.fit(X_train_comp_e, Y_train_comp)
    KNN.fit(X_train_comp_e, Y_train_comp)

    #Test classifiers
    ANN_pred = ANN.predict(X_test_comp_e)
    SVM_pred = SVM.predict(X_test_comp_e)
    SVM4_pred = SVM4.predict(X_test_comp_e)
    RFC_pred = RFC.predict(X_test_comp_e) 
    KNN_pred = KNN.predict(X_test_comp_e)

    from sklearn.metrics import confusion_matrix, roc_auc_score
    ANN_CM = confusion_matrix(Y_test,ANN_pred)
    SVM_CM = confusion_matrix(Y_test,SVM_pred)
    SVM4_CM = confusion_matrix(Y_test,SVM4_pred)
    RFC_CM = confusion_matrix(Y_test,RFC_pred)
    KNN_CM = confusion_matrix(Y_test,KNN_pred)

    ANN_ones = (ANN_CM[1,1] / float(ANN_CM[1,1]+ANN_CM[1,0]))*100
    ANN_zero = (ANN_CM[0,0] / float(ANN_CM[0,1]+ANN_CM[0,0]))*100

    SVM_ones = (SVM_CM[1,1] / float(SVM_CM[1,1]+SVM_CM[1,0]))*100
    SVM_zero = (SVM_CM[0,0] / float(SVM_CM[0,1]+SVM_CM[0,0]))*100
    
    SVM4_ones = (SVM4_CM[1,1] / float(SVM4_CM[1,1]+SVM4_CM[1,0]))*100
    SVM4_zero = (SVM4_CM[0,0] / float(SVM4_CM[0,1]+SVM4_CM[0,0]))*100

    RFC_ones = (RFC_CM[1,1] / float(RFC_CM[1,1]+SVM_CM[1,0]))*100
    RFC_zero = (RFC_CM[0,0] / float(RFC_CM[0,1]+RFC_CM[0,0]))*100

    KNN_ones = (KNN_CM[1,1] / float(KNN_CM[1,1]+KNN_CM[1,0]))*100
    KNN_zero = (KNN_CM[0,0] / float(KNN_CM[0,1]+KNN_CM[0,0]))*100



    print("Working on sample nr: ",  num_f, )


    ### COMPARE STUFF!
    print("###############")
    print("evaluating ", num_f, " number of features")
    print('{0:.3g}'.format((num_f/stop)*100), " % done")
    print("###############")
    print('Results on ', len(X_train), 'training samples and ', (len(X_test)), ' test samples.')
    print('ANN: 1: ', '{0:.3g}'.format(ANN_ones), '% | 0:', '{0:.3g}'.format(ANN_zero) ,'% | 1%0A:', '{0:.3g}'.format((ANN_ones+ANN_zero)/2), '% | 1&0 :' ,'{0:.3g}'.format(sklearn.metrics.accuracy_score(np.array(Y_test),np.array(ANN_pred))), '%', ' AURPC: ','{0:.3g}'.format(roc_auc_score(Y_test, ANN_pred, average='macro')))
    print('SVM: 1: ', '{0:.3g}'.format(SVM_ones), '% | 0:', '{0:.3g}'.format(SVM_zero) ,'% | 1%0A:', '{0:.3g}'.format((SVM_ones+SVM_zero)/2), '% | 1&0 :' ,'{0:.3g}'.format(sklearn.metrics.accuracy_score(np.array(Y_test),np.array(SVM_pred))), '%', ' AURPC: ','{0:.3g}'.format(roc_auc_score(Y_test, SVM_pred, average='macro')))
    print('SVM4: 1: ', '{0:.3g}'.format(SVM4_ones), '% | 0:', '{0:.3g}'.format(SVM4_zero) ,'% | 1%0A:', '{0:.3g}'.format((SVM4_ones+SVM4_zero)/2), '% | 1&0 :' ,'{0:.3g}'.format(sklearn.metrics.accuracy_score(np.array(Y_test),np.array(SVM4_pred))), '%', ' AURPC: ','{0:.3g}'.format(roc_auc_score(Y_test, SVM4_pred, average='macro')))
    print('RFC: 1: ', '{0:.3g}'.format(RFC_ones), '% | 0:', '{0:.3g}'.format(RFC_zero) ,'% | 1%0A:', '{0:.3g}'.format((RFC_ones+RFC_zero)/2), '% | 1&0 :' ,'{0:.3g}'.format(sklearn.metrics.accuracy_score(np.array(Y_test),np.array(RFC_pred))), '%', ' AURPC: ','{0:.3g}'.format(roc_auc_score(Y_test, RFC_pred, average='macro')))
    print('KNN: 1: ', '{0:.3g}'.format(KNN_ones), '% | 0:', '{0:.3g}'.format(KNN_zero) ,'% | 1%0A:', '{0:.3g}'.format((KNN_ones+KNN_zero)/2), '% | 1&0 :' ,'{0:.3g}'.format(sklearn.metrics.accuracy_score(np.array(Y_test),np.array(KNN_pred))), '%', ' AURPC: ','{0:.3g}'.format(roc_auc_score(Y_test, KNN_pred, average='macro')))
    
    
    # appending new results
    a[index] = roc_auc_score(Y_test, ANN_pred, average='macro')
    s[index] = roc_auc_score(Y_test, SVM_pred, average='macro')
    s4[index] = roc_auc_score(Y_test, SVM4_pred, average='macro')
    r[index] = roc_auc_score(Y_test, RFC_pred, average='macro')
    k[index] = roc_auc_score(Y_test, KNN_pred, average='macro')
    print("denna array: ", k[index])
    print("denna array: ", k[index])
    print("denna array: ", k[index])
    
    # Saving arrays to text (for saving in case of crash and for further use)
    np.savetxt("ann_roc.txt", np.array(a), fmt="%s")
    np.savetxt("svm_roc.txt", np.array(s), fmt="%s")
    np.savetxt("svm4_roc.txt", np.array(s4), fmt="%s")
    np.savetxt("rfc_roc.txt", np.array(r), fmt="%s")
    np.savetxt("knn_roc.txt", np.array(k), fmt="%s")
    #s[index] = roc_auc_score(Y_test, KNN_pred, average='macro')
    #np.savetxt("knn_new.txt", np.array(a), fmt="%s")
    
    print('This feature extraction took {} seconds'.format(time.time() - starttime))
    



# Arrays för multiprocessing, threadsafe (importante) and shared memory over processes

ann_array = multiprocessing.Array("f", [0] *array_size, lock=True)
svm_array = multiprocessing.Array("f", [0] *array_size, lock=True)
svm4_array = multiprocessing.Array("f", [0] *array_size, lock=True)
rfc_array = multiprocessing.Array("f", [0] *array_size, lock=True)
knn_array = multiprocessing.Array("f", [0] *array_size, lock=True)


# helper cause map function in python sucks
def process_helper(i):
    multiprocessing_func(i, ann_array, svm_array,svm4_array, rfc_array, knn_array)

# Multiprocessing Pool
from multiprocessing import Pool
if __name__ == '__main__':
    
    starttime = time.time()
    pool = Pool()
    pool.map(process_helper, range(1, array_size-1))
    pool.close()
    
    print('That took {} seconds'.format(time.time() - starttime))


#time.sleep(20)

##############################
#### Plotting the Results ####
##############################
    
# libraries and data
import matplotlib.pyplot as plt
import pandas as pd
 
# Make a data frame
df=pd.DataFrame({'x': range(start-1,stop-1, steps), 
                 'ann_roc_auc': ann_array, 
                 'rbf_svm_roc_auc': svm_array, 
                 'lin_svm_roc_auc': svm4_array, 
                 'rfc_roc_auc': rfc_array, 
                 'knn_roc_auc': knn_array, }) 
                 
# style
plt.figure(figsize=(25,15))
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
 
# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("ROC_AUC", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Num of features")
plt.ylabel("Score")

plt.xticks(np.arange(start-1,stop-1,steps))
#plt.figure(figsize=(12,8))
plt.savefig('weew.png')
plt.savefig('test.png', dpi = 400)
plt.savefig('eee.png', dpi = 1000)
