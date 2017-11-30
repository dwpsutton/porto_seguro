'''This script was inspired by snowdog's R kernel that builds an old school neural network 
   which scores quite well on the public LB relative to other NN approaches.
   https://www.kaggle.com/snowdog/old-school-nnet

   The idea is that after some pre-processing, a simpler network structure may generalize
   much better than a deep, complicated one. The network in this script has only 1 hidden layer
   with 35 neurons, uses some dropout, and trains for just 15 epochs. 
   Upsampling is also used, which seems to improve NN results. 
   
   We'll do a 5-fold split on the data, train 3 times on each fold and bag the predictions, then average
   the bagged predictions to get a submission. Increasing the number of training folds and the
   number of runs per fold would likely improve the results.
   
   The LB score is approximate because I haven't been able to get random seeding to properly
   make keras results consistent - any advice here would be much appreciated! 
'''

import numpy as np
np.random.seed(20)
import pandas as pd

from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import StratifiedKFold

'''Data loading & preprocessing
'''

X_train = pd.read_csv('../data/train.csv')
X_test = pd.read_csv('../data/test.csv')

#import sklearn.model_selection
#index_train, index_test= sklearn.model_selection.train_test_split( range(len(X_train.index)) ,
#                                                                   test_size=0.3,random_state=1)
##index_train= X_train.index[index_train]
##index_test= X_train.index[index_test]
#
#X_test, test_id = X_train.copy().loc[index_test,:].reset_index(drop=True), X_train.id.values[index_test]
#X_test.drop(['target','id'],inplace=True,axis=1)
#X_train, y_train, train_id = X_train.copy().loc[index_train,:].reset_index(drop=True), X_train.loc[index_train,'target'], X_train.id.values[index_train]
#X_train.drop(['target','id'],inplace=True,axis=1)

X_train, y_train, train_id = X_train.iloc[:,2:], X_train.target, X_train.id.values
X_test, test_id = X_test.iloc[:,1:], X_test.id

#OHE / some feature engineering adapted from the1owl kernel at:
#https://www.kaggle.com/the1owl/forza-baseline/code

#excluded columns based on snowdog's old school nn kernel at:
#https://www.kaggle.com/snowdog/old-school-nnet

X_train['negative_one_vals'] = np.sum((X_train==-1).values, axis=1)
X_test['negative_one_vals'] = np.sum((X_test==-1).values, axis=1)

to_drop = ['ps_car_11_cat', 'ps_ind_14', 'ps_car_11', 'ps_car_14', 'ps_ind_06_bin', 
           'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 
           'ps_ind_13_bin']

cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))
             & (not c in to_drop)]
             
X_train = X_train[cols_use]
X_test = X_test[cols_use]

one_hot = {c: list(X_train[c].unique()) for c in X_train.columns}

#note that this encodes the negative_one_vals column as well
for c in one_hot:
    if len(one_hot[c])>2 and len(one_hot[c]) < 105:
        for val in one_hot[c]:
            newcol = c + '_oh_' + str(val)
            X_train[newcol] = (X_train[c].values == val).astype(np.int)
            X_test[newcol] = (X_test[c].values == val).astype(np.int)
#        X_train.drop(labels=[c], axis=1, inplace=True)
#        X_test.drop(labels=[c], axis=1, inplace=True)

X_train = X_train.replace(-1, np.NaN)  # Get rid of -1 while computing interaction col
X_test = X_test.replace(-1, np.NaN)

X_train['ps_car_13_x_ps_reg_03'] = X_train['ps_car_13'] * X_train['ps_reg_03']
X_test['ps_car_13_x_ps_reg_03'] = X_test['ps_car_13'] * X_test['ps_reg_03']

X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)

#import pipe0_median as pipe0, sklearn.model_selection
#df_train = pd.read_csv('../data/train.csv')
#df_test = pd.read_csv('../data/test.csv')
#index_train, index_test= sklearn.model_selection.train_test_split( range(len(df_train.index)) ,
#                                                                  test_size=0.3,random_state=1)
#
#df_test= df_train.loc[index_test,:].reset_index(drop=True)
#y_test= df_test.target.values
#df_train= df_train.loc[index_train,:].reset_index(drop=True)
#y_train= df_train.target.values
#train_id= df_train.id.values
#test_id= df_test.id.values
#
#X_trainb, y_train, X_testb, y_test= pipe0.run_pipe0( df_train.copy(), df_test.copy() )
##temp= {}
##for i in range(np.shape(X_trainb)[1]):
##    temp['x'+str(i)]= X_trainb[:,i]
##X_train= pd.DataFrame(temp, columns= ['x'+str(i) for i in range(np.shape(X_trainb)[0])] )
##temp= {}
##for i in range(np.shape(X_testb)[1]):
##    temp['x'+str(i)]= X_testb[:,i]
##X_test= pd.DataFrame(temp, columns= ['x'+str(i) for i in range(np.shape(X_testb)[0])])
##print 'read data...'

'''Gini scoring function
'''

#gini scoring function from kernel at: 
#https://www.kaggle.com/tezdhar/faster-gini-calculation
def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)

'''5-fold neural network training 
'''

K = 10 #number of folds
runs_per_fold = 3 #bagging on each fold

cv_ginis = []
val_y_preds = np.zeros((np.shape(X_train)[0],K))
y_preds = np.zeros((np.shape(X_test)[0],K))

kfold = StratifiedKFold(n_splits = K, 
                            random_state = 100, 
                            shuffle = True)    

for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
          
    #upsampling adapted from kernel: 
    #https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(y_train_f == 1))
    
    # Add positive examples
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)
    
    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    
    #track oof bagged prediction for cv scores <-- nb: this is NOT bagging.  There is no bootstrap.
    val_preds = 0
    
    for j in range(runs_per_fold):
    
        NN=Sequential()
        NN.add(Dense(35,activation='relu',input_dim=np.shape(X_train_f)[1]))
        NN.add(Dropout(0.3))
        NN.add(Dense(1,activation='sigmoid'))
        
        NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        set_random_seed(1000*i+j)
            
        NN.fit(X_train_f.values, y_train_f.values,epochs=15,
               batch_size=2048, verbose=0)
         
        val_gini = gini_normalizedc(y_val_f.values, NN.predict(X_val_f.values)[:,0])   
        print ('\nFold %d Run %d Results *****' % (i, j))
        print ('Validation gini: %.5f\n' % (val_gini))
        
        val_preds += NN.predict(X_val_f.values)[:,0] / runs_per_fold
        val_y_preds[outf_ind,i] += val_preds
        y_preds[:,i] += NN.predict(X_test.values)[:,0] / runs_per_fold
        
    cv_ginis.append(val_gini)
    print ('\nFold %i prediction cv gini: %.5f\n' %(i,val_gini))
    
print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
y_pred_final = np.mean(y_preds, axis=1)
val_y_pred_final = np.mean(val_y_preds, axis=1)

pd.DataFrame({'id':train_id,
             'target': val_y_pred_final},
             columns=['id','target']
             ).to_csv('internals/keras_oob.csv',index=False)

df_sub = pd.DataFrame({'id' : test_id, 
                       'target' : y_pred_final},
                       columns = ['id','target'])
df_sub.to_csv('internals/keras_test.csv', index=False)
