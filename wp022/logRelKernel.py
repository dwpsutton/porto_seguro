import pandas as pd, numpy as np
import sklearn.model_selection, sklearn.linear_model
from datetime import datetime,timedelta


def missing_value(df):
    col = df.columns
    for i in col:
        if df[i].isnull().sum()>0:
            df[i].fillna(df[i].mode()[0],inplace=True)

def category_type(df):
    col = df.columns
    for i in col:
        if df[i].nunique()<=104:
            df[i] = df[i].astype('category')

def outlier(df,columns):
    for i in columns:
        quartile_1,quartile_3 = np.percentile(df[i],[25,75])
        quartile_f,quartile_l = np.percentile(df[i],[1,99])
        IQR = quartile_3-quartile_1
        lower_bound = quartile_1 - (1.5*IQR)
        upper_bound = quartile_3 + (1.5*IQR)
#        print(lower_bound,upper_bound)
#        print(quartile_f,quartile_l)

        df[i].loc[df[i] < lower_bound] = quartile_f
        df[i].loc[df[i] > upper_bound] = quartile_l

def OHE(df1,df2,column):
    cat_col = column
    #cat_col = df.select_dtypes(include =['category']).columns
    len_df1 = df1.shape[0]
    
    df = pd.concat([df1,df2],ignore_index=True)
    c2,c3 = [],{}
    
    print('Categorical feature',len(column))
    for c in cat_col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df, prefix=c3, columns=c2,drop_first=True)
#    df2 = pd.get_dummies(df2, prefix=c3, columns=c2,drop_first=True)

    df1 = df.loc[:len_df1-1]
    df2 = df.loc[len_df1:]
#    print('Train',df1.shape)
#    print('Test',df2.shape)
    return df1,df2


def run_pipe4(train,test):

    ps_cal = train.columns[train.columns.str.startswith('ps_calc')]
    train = train.drop(ps_cal,axis =1)
    test = test.drop(ps_cal,axis=1)
    train.shape


    missing_value(train)
    missing_value(test)

    category_type(train)
    category_type(test)

    cat_col = [col for col in train.columns if '_cat' in col]


    bin_col = [col for col in train.columns if 'bin' in col]

    tot_cat_col = list(train.select_dtypes(include=['category']).columns)

    other_cat_col = [c for c in tot_cat_col if c not in cat_col+ bin_col]
    other_cat_col

    num_col = [c for c in train.columns if c not in tot_cat_col]
    num_col.remove('id')


    outlier(train,num_col)
    outlier(test,num_col)


    print 'dave: ',len(train.columns),len(test.columns)
    
    train1,test1 = OHE(train,test,tot_cat_col)

    X = train1.drop(['target','id'],axis=1)
    y = train1['target'].astype('category')
    y_test= test1.target.values
    X_test = test1.drop(['target','id'],axis=1)
    return X,y,X_test,y_test


def bag_classifier(base,PIPELINE,df_train,df_test):
    import bagging_me
    tc= datetime.now()
    clf= bagging_me.BaggingClassifier(base,
                                      n_estimators=32,
                                      oob_score=True,
                                      max_features=1.0,
                                      max_samples=0.8,
                                      random_state=1,
                                      n_jobs=-1
                                      )
    X_train, y_train, X_test, y_test= PIPELINE( df_train.copy(),df_test.copy() )
    clf.fit(X_train,y_train,sample_weight=None)
    oob= clf.oob_decision_function_[:,1]
    y_pred= clf.predict_proba(X_test)[:,1]
    print 'Training took: ',(datetime.now() - tc).total_seconds()
    return oob, y_pred

def model():

    df_train= pd.read_csv('../data/train.csv',na_values=-1)
    df_test= pd.read_csv('../data/test.csv',na_values=-1)

    index_train, index_test= sklearn.model_selection.train_test_split( range(len(df_train.index)) ,
                                                                      test_size=0.3,random_state=1)

    df_test= df_train.loc[index_test,:].reset_index(drop=True)
    y_test= df_test.target.values
    df_train= df_train.loc[index_train,:].reset_index(drop=True)
    y_train= df_train.target.values

    clf2= sklearn.linear_model.LogisticRegression(C=0.01,
                                              class_weight='balanced',
                                              penalty='l2')
    tc= datetime.now()
#    print 'Running processing pipe4...'
#    X_train, y_train, X_test, y_test= run_pipe4( df_train.copy(),df_test.copy() )
#    clf2.fit(X_train,y_train)
#    ptest= clf2.predict_proba(X_test)[:,1]
    oob, ptest= bag_classifier(clf2,run_pipe4,df_train,df_test)
    print 'Training took: ',(datetime.now() - tc).total_seconds()
    print 'OOB Gini score= ',2.*sklearn.metrics.roc_auc_score(y_train,oob)-1.
    print 'Test Gini score= ',2.*sklearn.metrics.roc_auc_score(y_test,ptest)-1.
    return pd.DataFrame({'id':df_train.id.values,'target':oob},
                        columns=['id','target']
                        ),pd.DataFrame({'id':df_test.id.values,'target':ptest},
                                       columns=['id','target']
                                       )


if __name__=='__main__':
    cv,test= model()
    cv.to_csv('internals/LRbase_oob.csv',index=False)
    test.to_csv('internals/LRBase_test.csv',index=False)


##Random search
#logreg = LogisticRegression(class_weight='balanced')
#param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1]}
#clf = RandomizedSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=3)
#clf.fit(X,y)
#print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))