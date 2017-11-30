import numpy as np, pandas as pd
from datetime import datetime,timedelta
import sklearn.model_selection,sklearn.metrics, pipe0
from sklearn.cross_validation import StratifiedKFold
import rgf.sklearn
from multiprocessing import Process,Queue


rgf_params= {'max_leaf': 3000,
            'test_interval':100,
            'l2':0.01,
            'verbose':0,
            'learning_rate':0.5,
            'normalize':False,
            'loss':'Log',
            'algorithm':'RGF',
            'min_samples_leaf':300
            }


def train_fold(X_train,y_train,X_val,X_test):
    #train model instance on a fold
    rgfm= rgf.sklearn.RGFClassifier(**rgf_params)
    rgfm.fit(X_train,y_train)
    y_pred= rgfm.predict_proba(X_val)[:,1]
    y_test= rgfm.predict_proba(X_test)[:,1]
    return y_pred, y_test, rgfm


def train_fold_subprocess(q,X_train,y_train,X_val,X_test):
    y_pred, y_test, model= train_fold(X_train,y_train,X_val,X_test)
    q.put( [y_pred, y_test, model._estimators[0]._latest_model_loc] )
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
        X_train, y_train, X_holdout, y_holdout= pipe0.run_pipe0( df_train.copy().loc[train_idx,:],
                                                                     df_train.copy().loc[val_idx,:] )
        _,_, X_test, y_test= pipe0.run_pipe0( df_train.copy(),df_test.copy() )
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
            print 'fold '+str(i)+', model: ',yinfo[2]
            p[i].join()
    S_test= S_test_i.mean(axis=1)
    return S_train, S_test

def validate_sub_models(df_train,df_test):
    n_folds=5
    folds = StratifiedKFold(df_train.target.values,n_folds=n_folds, shuffle=True, random_state=2016)
    S_train= np.zeros(len(df_train.index))
    S_test_i= np.zeros( [len(df_test.index), n_folds] )
    
    folds_Info=[]
    for i, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train, X_holdout, y_holdout= pipe0.run_pipe0( df_train.copy().loc[train_idx,:],
                                                                df_train.copy().loc[val_idx,:] )
        _,_, X_test, y_test= pipe0.run_pipe0( df_train.copy(),df_test.copy() )
        foldInfo= {}
        foldInfo['X_train']= X_train
        foldInfo['y_train']= y_train
        foldInfo['X_holdout']= X_holdout
        foldInfo['y_holdout']= y_holdout
        foldInfo['X_test']= X_test
        foldInfo['y_test']= y_test
        folds_Info.append(foldInfo)
    
    for nn in range(18,30):
        nn_str= '%02d' % (nn+1)
        model_names= ['output_0.01_300leaf_run2/models/d0331abb-5465-4cc2-907c-2d0eb0184c491.model-'+nn_str,
                      'output_0.01_300leaf_run2/models/751aceed-0e2b-4047-8204-e8623a678ff61.model-'+nn_str,
                      'output_0.01_300leaf_run2/models/9c1779f2-8429-48fc-b40e-2dba24eba5731.model-'+nn_str,
                      'output_0.01_300leaf_run2/models/f8738eac-b191-47a1-ad40-c0368534ddae1.model-'+nn_str,
                      'output_0.01_300leaf_run2/models/1b10cdf5-7d51-444e-be00-ce56a5c900c41.model-'+nn_str]
#        model_names=['output_0.1/models/a4803b23-600a-4af0-92a4-ede88ce22e541.model-'+nn_str,
#                     'output_0.1/models/c7401c98-600c-4723-957c-be2326f16b561.model-'+nn_str,
#                     'output_0.1/models/0466b9ce-cd97-4720-a740-535ef4b5a7321.model-'+nn_str,
#                     'output_0.1/models/1e314d01-5116-46d3-9464-2cac49f18eb21.model-'+nn_str,
#                     'output_0.1/models/6d23b585-6f01-4a72-ac95-5de98b2001331.model-'+nn_str]

#        model_names=['output_1.0/models/8861b0ce-563e-471e-9571-24eea5d296661.model-'+nn_str,
#                     'output_1.0/models/602fab9b-9d5c-4407-b56a-e04b0908c4a01.model-'+nn_str,
#                     'output_1.0/models/6c9a01b7-7017-4a43-9464-4625b2f7fa681.model-'+nn_str,
#                     'output_1.0/models/fc6743bb-fe52-4bc9-831f-f6b18e1821d81.model-'+nn_str,
#                     'output_1.0/models/26bde7b6-73af-45a0-a81c-3c994079158e1.model-'+nn_str]

#        model_names=['/tmp/rgf/8c8c261f-ac3a-42a0-8f97-e84678eaa61f1.model-'+nn_str,
#                      '/tmp/rgf/0fdd8bf8-7bc4-4a3f-b33c-bc67654abc8f1.model-'+nn_str,
#                      '/tmp/rgf/bb259424-d5d7-4bba-aefb-b5c5799dd3661.model-'+nn_str,
#                      '/tmp/rgf/dfb58c59-1cb0-4268-9d74-1d2a6d6e29af1.model-'+nn_str,
#                      '/tmp/rgf/5329f494-b8e2-4a08-b953-eb0a4349ac261.model-'+nn_str]

#['/tmp/rgf/49bd977c-551e-4d1d-a174-dc31ba1672e31.model-'+nn_str,
#                      '/tmp/rgf/c5928c8b-5169-4943-938a-b2057fc154511.model-'+nn_str,
#                      '/tmp/rgf/bcc80f22-1f4e-4276-84dc-1e647bfffd481.model-'+nn_str,
#                      '/tmp/rgf/b65a70ce-c85c-483e-9ed7-421aef44fa8c1.model-'+nn_str,
#                      '/tmp/rgf/9103c554-869a-44e9-adb3-7f802da68fb81.model-'+nn_str]
        S_train= np.zeros(len(df_train.index))
        for i, (train_idx, val_idx) in enumerate(folds):
            # Get this fold's data
#            print 'Initiating fold ',i
#            X_train, y_train, X_holdout, y_holdout= pipe0.run_pipe0( df_train.copy().loc[train_idx,:],
#                                                                    df_train.copy().loc[val_idx,:] )
#            _,_, X_test, y_test= pipe0.run_pipe0( df_train.copy(),df_test.copy() )
            X_train= folds_Info[i]['X_train']
            y_train= folds_Info[i]['y_train']
            X_holdout= folds_Info[i]['X_holdout']
            y_holdout= folds_Info[i]['y_holdout']
            X_test= folds_Info[i]['X_test']
            y_test= folds_Info[i]['y_test']
            rgfm= rgf.sklearn.RGFClassifier(**rgf_params)
            rgfm._estimators= [rgf.sklearn._RGFBinaryClassifier(**rgf_params)]
            rgfm._estimators[0]._latest_model_loc= model_names[i]
            rgfm._fitted = True
            rgfm._n_features = np.shape(X_holdout)[1]
            rgfm._n_classes= 2
            rgfm._estimators[0]._fitted= True
            rgfm._estimators[0]._n_features = np.shape(X_holdout)[1]
            rgfm._estimators[0]._n_classes= 2
#            print rgfm._estimators[0]._latest_model_loc
            S_train[val_idx]= rgfm.predict_proba(X_holdout)[:,1]
        print 'Model '+nn_str+' Gini score= ',2.*sklearn.metrics.roc_auc_score(df_train.target.values,S_train)-1.

def score_optimal_model(df_train,df_test):
    n_folds=5
    folds = StratifiedKFold(df_train.target.values,n_folds=n_folds, shuffle=True, random_state=2016)
    S_train= np.zeros(len(df_train.index))
    S_test_i= np.zeros( [len(df_test.index), n_folds] )
    
    nn=19 #16
    nn_str= '%02d' % (nn+1)
    model_names= ['output_0.01_300leaf/models/b7e3d140-7a3b-4ec4-8488-aefc1a3531211.model-'+nn_str,
                  'output_0.01_300leaf/models/3d96ef6d-60d6-4a5d-97ee-c4ee4e0c4cba1.model-'+nn_str,
                  'output_0.01_300leaf/models/2f25b838-7fb7-4a51-8ae0-3f076ab30d341.model-'+nn_str,
                  'output_0.01_300leaf/models/d15b02c1-751e-4911-a581-24e3c48744621.model-'+nn_str,
                  'output_0.01_300leaf/models/67f5ca53-15ff-4485-9c6d-bb010cf860cb1.model-'+nn_str]
#    model_names=['output_0.1/models/a4803b23-600a-4af0-92a4-ede88ce22e541.model-'+nn_str,
#                     'output_0.1/models/c7401c98-600c-4723-957c-be2326f16b561.model-'+nn_str,
#                     'output_0.1/models/0466b9ce-cd97-4720-a740-535ef4b5a7321.model-'+nn_str,
#                     'output_0.1/models/1e314d01-5116-46d3-9464-2cac49f18eb21.model-'+nn_str,
#                     'output_0.1/models/6d23b585-6f01-4a72-ac95-5de98b2001331.model-'+nn_str]
#    model_names= ['output_0.01/models/49bd977c-551e-4d1d-a174-dc31ba1672e31.model-'+nn_str,
#                  'output_0.01/models/c5928c8b-5169-4943-938a-b2057fc154511.model-'+nn_str,
#                  'output_0.01/models/bcc80f22-1f4e-4276-84dc-1e647bfffd481.model-'+nn_str,
#                  'output_0.01/models/b65a70ce-c85c-483e-9ed7-421aef44fa8c1.model-'+nn_str,
#                  'output_0.01/models/9103c554-869a-44e9-adb3-7f802da68fb81.model-'+nn_str]
    S_train= np.zeros(len(df_train.index))
    S_test_i= np.zeros([df_test.id.count(),n_folds])
    for i, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train, X_holdout, y_holdout= pipe0.run_pipe0( df_train.copy().loc[train_idx,:],
                                                                 df_train.copy().loc[val_idx,:] )
        _,_, X_test, y_test= pipe0.run_pipe0( df_train.copy(),df_test.copy() )
        rgfm= rgf.sklearn.RGFClassifier(**rgf_params)
        rgfm._estimators= [rgf.sklearn._RGFBinaryClassifier(**rgf_params)]
        rgfm._estimators[0]._latest_model_loc= model_names[i]
        rgfm._fitted = True
        rgfm._n_features = np.shape(X_holdout)[1]
        rgfm._n_classes= 2
        rgfm._estimators[0]._fitted= True
        rgfm._estimators[0]._n_features = np.shape(X_holdout)[1]
        rgfm._estimators[0]._n_classes= 2
            
        S_train[val_idx]= rgfm.predict_proba(X_holdout)[:,1]
        Stesti= rgfm.predict_proba(X_test)[:,1]
        S_test_i[:,i]= pd.DataFrame({'myd':Stesti},columns=['myd']).myd.rank()
        print 'scored using classifier '+str(i+1) + ' of '+str(len(folds))
    S_test= np.median(S_test_i,axis=1) / float(np.shape(X_test)[0])
    print 'Model '+nn_str+' Gini score= ',2.*sklearn.metrics.roc_auc_score(df_train.target.values,S_train)-1.
    return S_test


def main():
    df_train= pd.read_csv('../data/train.csv')
    df_test= pd.read_csv('../data/test.csv')
    tc=datetime.now()
    y_pred, y_test= cross_validate_train(5,df_train,df_test,distribute=True)
    print 'Training took: ',(datetime.now() - tc).total_seconds()
    print 'Gini score= ',2.*sklearn.metrics.roc_auc_score(df_train.target.values,y_pred)-1.
    pd.DataFrame( {'id':df_train['id'].values,'target': y_pred}, columns= ['id','target']).to_csv('rgf_validation_ scores.csv',index=False)
    pd.DataFrame( {'id': df_test['id'].values,'target': y_test}, columns= ['id','target']).to_csv('rgf_blind_scores.csv',index=False)
    return None

if __name__=='__main__':
    main()
