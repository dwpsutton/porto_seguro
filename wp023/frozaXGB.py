import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from multiprocessing import *
import copy


def main(train,test):
    col = [c for c in train.columns if c not in ['id','target']]
    print(len(col))
    col = [c for c in col if not c.startswith('ps_calc_')]
    print(len(col))

    train = train.replace(-1, np.NaN)
    d_median = train.median(axis=0)
    d_mean = train.mean(axis=0)
    train = train.fillna(-1)
    one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}


    def transform_df(df):
        df = pd.DataFrame(df)
        dcol = [c for c in df.columns if c not in ['id','target']]
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
        for c in dcol:
            if '_bin' not in c: #standard arithmetic
                df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
                df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
        #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
        #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
        #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
        #df[c+str('_exp')] = np.exp(df[c].values) - 1
        for c in one_hot:
            if len(one_hot[c])>2 and len(one_hot[c]) < 7:
                for val in one_hot[c]:
                    df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
        return df

#    def transform_df(df):
#        print('Init Shape: ', df.shape)
#        p = Pool(cpu_count())
#        df = p.map(transform_df, np.array_split(df, cpu_count()))
#        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
#        p.close(); p.join()
#        print('After Shape: ', df.shape)
#        return df

    def gini(y, pred):
        fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
        g = 2 * metrics.auc(fpr, tpr) -1
        return g

    def gini_xgb(pred, y):
        y = y.get_label()
        return 'gini', gini(y, pred)

    def gini_lgb(preds, dtrain):
        y = list(dtrain.get_label())
        score = gini(y, preds) / gini(y, y)
        return 'gini', score, True


    params = {'eta': 0.09, 'max_depth': 4, 'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 0.77, 'scale_pos_weight': 1.6, 'gamma': 10, 'reg_alpha': 8, 'reg_lambda': 1.3, 'eval_metric': 'auc', 'seed': 99, 'silent': True, 'n_jobs':8}
    x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)

    x1 = transform_df(x1)
    x2 = transform_df(x2)
    test = transform_df(test)

    col = [c for c in x1.columns if c not in ['id','target']]
    col = [c for c in col if not c.startswith('ps_calc_')]
    print(x1.values.shape, x2.values.shape)

    #remove duplicates just in case
    tdups = transform_df(train)
    dups = tdups[tdups.duplicated(subset=col, keep=False)]

    x1 = x1[~(x1['id'].isin(dups['id'].values))]
    x2 = x2[~(x2['id'].isin(dups['id'].values))]
    print(x1.values.shape, x2.values.shape)

    y1 = x1['target']
    y2 = x2['target']
    x1 = x1[col]
    x2 = x2[col]

    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 500,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=20, early_stopping_rounds=20)
    test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+10)
    test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
    test[['id','target']].to_csv('froza/xgb_submission.csv', index=False, float_format='%.5f')

#    #LightGBM
#    params = {'learning_rate': 0.02, 'max_depth': 4, 'boosting': 'gbdt', 'objective': 'binary', 'max_bin': 10, 'subsample': 0.8, 'subsample_freq': 10, 'colsample_bytree': 0.8, 'min_child_samples': 500, 'metric': 'auc', 'is_training_metric': False, 'seed': 99, 'n_jobs':8}
#    model2 = lgb.train(params, lgb.Dataset(x1, label=y1), 1000, lgb.Dataset(x2, label=y2), verbose_eval=50, feval=gini_lgb, early_stopping_rounds=200)
#    test['target'] = model2.predict(test[col], num_iteration=model2.best_iteration)
#    test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
#    test[['id','target']].to_csv('froza/lgb_submission.csv', index=False, float_format='%.5f')


def mainCV(train,test):
    col = [c for c in train.columns if c not in ['id','target']]
    print(len(col))
    col = [c for c in col if not c.startswith('ps_calc_')]
    print(len(col))
    
    train = train.replace(-1, np.NaN)
    d_median = train.median(axis=0)
    d_mean = train.mean(axis=0)
    train = train.fillna(-1)
    one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}
    
    def transform_df(df):
        df = pd.DataFrame(df)
        dcol = [c for c in df.columns if c not in ['id','target']]
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
        for c in dcol:
            if '_bin' not in c: #standard arithmetic
                df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
                df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
        for c in one_hot:
            if len(one_hot[c])>2 and len(one_hot[c]) < 7:
                for val in one_hot[c]:
                    df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
        return df

    def gini(y, pred):
        fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
        g = 2 * metrics.auc(fpr, tpr) -1
        return g

    def gini_xgb(pred, y):
        y = y.get_label()
        return 'gini', gini(y, pred)

    def gini_lgb(preds, dtrain):
        y = list(dtrain.get_label())
        score = gini(y, preds) / gini(y, y)
        return 'gini', score, True


    params = {'eta': 0.09, 'max_depth': 4, 'objective': 'binary:logistic', 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 0.77, 'scale_pos_weight': 1.6, 'gamma': 10, 'reg_alpha': 8, 'reg_lambda': 1.3, 'eval_metric': 'auc', 'seed': 99, 'silent': True, 'n_jobs':8}
    
    paramsLGB = {'learning_rate': 0.02, 'max_depth': 4, 'boosting': 'gbdt', 'objective': 'binary', 'max_bin': 10, 'subsample': 0.8, 'subsample_freq': 10, 'colsample_bytree': 0.8, 'min_child_samples': 500, 'metric': 'auc', 'is_training_metric': False, 'seed': 99, 'nthread':8, 'verbosity':-1}

    test = transform_df(test)
    otest= pd.DataFrame({'id':test.id.values,'target':np.zeros(test.id.count())},columns=['id','target'])
    otest2= pd.DataFrame({'id':test.id.values,'target':np.zeros(test.id.count())},columns=['id','target'])

    K=10
    kf = KFold(n_splits = K, random_state = 2, shuffle = True)
    oof= np.zeros(train.target.count())
    oof2= np.zeros(train.target.count())
    for i, (train_index, test_index) in enumerate(kf.split(train)):
#        x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)
        x1= copy.deepcopy(train.loc[train_index,:])
        x2= copy.deepcopy(train.loc[test_index,:])

        x1 = transform_df(x1)
        x2 = transform_df(x2)
        
        col = [c for c in x1.columns if c not in ['id','target']]
        col = [c for c in col if not c.startswith('ps_calc_')]
        print(x1.values.shape, x2.values.shape)
        
#        remove duplicates just in case
#        tdups = transform_df(train)
#        dups = tdups[tdups.duplicated(subset=col, keep=False)]
#
#        x1 = x1[~(x1['id'].isin(dups['id'].values))]
#        x2 = x2[~(x2['id'].isin(dups['id'].values))]
#        print(x1.values.shape, x2.values.shape)

        y1 = x1['target']
        y2 = x2['target']
        x1 = x1[col]
        x2 = x2[col]
        
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), 500,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=20, early_stopping_rounds=20)
        oof[test_index] = model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit+10)
        otest['target'] += model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+10) / K

         #LGB model
        model2 = lgb.train(paramsLGB, lgb.Dataset(x1, label=y1), 1000, lgb.Dataset(x2, label=y2), verbose_eval=50, feval=gini_lgb, early_stopping_rounds=200)
        oof2[test_index] = model2.predict(x2, num_iteration=model2.best_iteration)
        otest2['target'] += model2.predict(test[col], num_iteration=model2.best_iteration) / K
    
    otest['target'] = (np.exp(otest['target'].values) - 1.0).clip(0,1)
    oof = (np.exp(oof) - 1.0).clip(0,1)
    otest2['target'] = (np.exp(otest2['target'].values) - 1.0).clip(0,1)
    oof2 = (np.exp(oof2) - 1.0).clip(0,1)

    #Blending
    out= pd.DataFrame({'id':test.id.values,'target':np.zeros(test.id.count())},columns=['id','target'])
    out.loc[:,'target']= (otest['target'].values * 0.4)  + (otest2.target.values * 0.6)
    out['target']= (np.exp(out['target'].values) - 1.0).clip(0,1)
    otest= out
    oof= (oof * 0.4)  + (oof2 * 0.6)
    oof= (np.exp(oof) - 1.0).clip(0,1)

    print 'Gini CV: ',2 * metrics.roc_auc_score(train.target.values, oof) -1
    return oof, otest
        
#        #LightGBM
#        params = {'learning_rate': 0.02, 'max_depth': 4, 'boosting': 'gbdt', 'objective': 'binary', 'max_bin': 10, 'subsample': 0.8, 'subsample_freq': 10, 'colsample_bytree': 0.8, 'min_child_samples': 500, 'metric': 'auc', 'is_training_metric': False, 'seed': 99, 'n_jobs':8}
#        model2 = lgb.train(params, lgb.Dataset(x1, label=y1), 1000, lgb.Dataset(x2, label=y2), verbose_eval=50, feval=gini_lgb, early_stopping_rounds=200)
#        test['target'] = model2.predict(test[col], num_iteration=model2.best_iteration)
#        test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
#        test[['id','target']].to_csv('froza/lgb_submission.csv', index=False, float_format='%.5f')



if __name__=='__main__':
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    oof, ptest= mainCV(train,test)
    ptest.to_csv('internals_test/froza_test.csv',index=False)
    pd.DataFrame({'id':train.id.values,'target':oof},columns=['id','target']).to_csv('internals_test/froza_oof.csv',index=False)

#    df1 = pd.read_csv('froza/xgb_submission.csv')
#    df2 = pd.read_csv('froza/lgb_submission.csv')
#    df2.columns = [x+'_' if x not in ['id'] else x for x in df2.columns]
#    blend = pd.merge(df1, df2, how='left', on='id')
#    for c in df1.columns:
#        if c != 'id':
#            blend[c] = (blend[c] * 0.4)  + (blend[c+'_'] * 0.6)
#    blend = blend[df1.columns]
#    blend['target'] = (np.exp(blend['target'].values) - 1.0).clip(0,1)
#    blend.to_csv('froza/blend1.csv', index=False, float_format='%.5f')
