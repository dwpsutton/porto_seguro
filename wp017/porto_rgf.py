import numpy as np, pandas as pd
from datetime import datetime,timedelta
import sklearn.model_selection,sklearn.metrics, pipe1
from sklearn.cross_validation import StratifiedKFold
import rgf.sklearn
from multiprocessing import Process,Queue


rgf_params= {'max_leaf': 8000,
            'test_interval':100,
            'l2':0.1,
            'verbose':0
            }


def train_fold(X_train,y_train,X_val,X_test):
    #train model instance on a fold
    rgfm= rgf.sklearn.RGFClassifier(**rgf_params)
    rgfm.fit(X_train,y_train)
    y_pred= rgfm.predict_proba(X_val)[:,1]
    y_test= rgfm.predict_proba(X_test)[:,1]
    return y_pred, y_test


def train_fold_subprocess(q,X_train,y_train,X_val,X_test):
    y_pred, y_test= train_fold(X_train,y_train,X_val,X_test)
    q.put( [y_pred, y_test] )
    return None

def cross_validate_train(n_folds,df_train,df_test,distribute=False):
    q= []
    p= []
    folds = StratifiedKFold(df_train.target.values,n_folds=n_folds, shuffle=True, random_state=2016)
    S_train= np.zeros(len(df_train.index))
    S_test_i= np.zeros( [len(df_test.index), n_folds] )
    for i, (train_idx, val_idx) in enumerate(folds):
        # Get this fold's data
        print 'Initiating fold ',i
        X_train, y_train, X_holdout, y_holdout= pipe1.run_pipe1( df_train.copy().loc[train_idx,:],
                                                                     df_train.copy().loc[val_idx,:] )
        _,_, X_test, y_test= pipe1.run_pipe1( df_train.copy(),df_test.copy() )
        if distribute:
            q.append( Queue() )
            p.append( Process(target=train_fold_subprocess,args=(q[i],X_train,y_train,X_holdout,X_test,)) )
            p[i].start()
        else:
            S_train[val_idx], yti= train_fold(X_train,y_train,X_holdout,X_test)
            S_test_i[:,i]= yti
    if distribute:
        for i, (train_idx, val_idx) in enumerate(folds):
            yinfo= q[i].get()
            S_train[val_idx]= yinfo[0]
            S_test_i[:,i]= yinfo[1]
            p[i].join()
    S_test= S_test_i.mean(axis=1)
    return S_train, S_test


def main():
    df_train= pd.read_csv('../data/train.csv')
    df_test= pd.read_csv('../data/test.csv')
    tc=datetime.now()
    y_pred, y_test= cross_validate_train(5,df_train,df_test,distribute=True)
    print 'Training took: ',(datetime.now() - tc).total_seconds()
    print 'Gini score= ',2.*sklearn.metrics.roc_auc_score(df_train.target.values,y_pred)-1.
    pd.DataFrame( {'id':df_train['id'].values,'target': y_pred}, columns= ['id','target']).to_csv('rgf_scores_train.csv',index=False)
    pd.DataFrame( {'id': df_test['id'].values,'target': y_test}, columns= ['id','target']).to_csv('rgf_blind_scores.csv',index=False)
    return None

if __name__=='__main__':
    main()
