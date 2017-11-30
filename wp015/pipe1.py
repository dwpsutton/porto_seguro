		import pandas as pd, numpy as np
import matplotlib.pyplot as plt


class prevalence_vectorizer:
    def __init__(self,field_list,target_col):
        self.field_list= field_list
        self.target_col= target_col
        self.vectorizer= {}
        for f in field_list:
            self.vectorizer[f]= {}
        return None
    
    def beta_prior_ratio(self,x):
        return float(np.sum(x[self.target_col]) + self.alpha) / (len(x[self.target_col]) + self.beta)
    
    def train(self,df):
        #
        pr_prev= df[self.target_col].mean()
        prior_obs= 20.
        self.alpha= pr_prev*prior_obs
        self.beta= prior_obs - self.alpha
        #
        data_fields= map(lambda x: str(x),df.columns)
        for f in self.field_list:
            if f in data_fields:
                self.vectorizer[f]= {}
                grps= df.groupby(f).apply(self.beta_prior_ratio)
                for g in grps.index.values:
                    self.vectorizer[f][g]= grps[g]
            else:
                print 'Warning: field '+f+' not in train data'
        return None
    
    def parse(self,vec,field_name):
        if field_name in self.field_list:
            return map(lambda x: self.vectorizer[field_name][x] if x in self.vectorizer[field_name] else pr_prev,vec)
        else:
            print field_name + ' not in vectorizer. Available fields: '+ self.field_list
            raise Exception


def transform_df(indf):
        
    # Add some extra features
    indf = indf.replace(-1, np.NaN)
    d_median = indf.median(axis=0)
    d_mean = indf.mean(axis=0)
    indf = indf.fillna(-1)
    
    indf = pd.DataFrame(indf)
    dcol = [c for c in indf.columns if c not in ['id','target']]
    indf['ps_car_13_x_ps_reg_03'] = indf['ps_car_13'] * indf['ps_reg_03']
    #indf['ps_reg_03_x_ps_car_14'] = indf['ps_car_14'] * indf['ps_reg_03']

    #indf['multireg']= indf['ps_reg_01'].values * indf['ps_reg_03'].values * indf['ps_reg_02'].values
    #indf['multireg']= indf['ps_ind_03'].values * indf['ps_ind_15'].values
    #     indf['ps_reg_01_x_ps_car_02_cat'] = str(indf['ps_reg_01']) + str(indf['ps_car_02_cat'])
    #     indf['ps_reg_01_x_ps_car_04_cat'] = str(indf['ps_reg_01']) + str(indf['ps_car_04_cat'])
    indf['negative_one_vals'] = np.sum((indf[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            indf[c+str('_median_range')] = (indf[c].values > d_median[c]).astype(np.int)
            indf[c+str('_mean_range')] = (indf[c].values > d_mean[c]).astype(np.int)
    return indf


def recon(reg):
    integer = int(np.round((40*reg)**2))
    for a in range(27):
        if (integer - a) % 27 == 0:
            A = a
    #     M = (integer - A)//31
    return A


def run_pipe1(train_df,test_df):


    # Group columns by type
    colnames= map(lambda x: str(x),train_df.columns.values)
    id_col= colnames[0]
    target_col= colnames[1]
    cat_cols= filter(lambda x: '_cat' in x,colnames)
    bin_cols= filter(lambda x: '_bin' in x,colnames)
    num_cols= filter(lambda x: ('_cat' not in x) and ('_bin' not in x) and x not in [id_col,target_col] ,
    colnames)

    # Cut bad fields, based on kernel discussions
    bin_cols.remove('ps_ind_12_bin')
    train_df.drop('ps_ind_12_bin',axis=1,inplace=True)

    # Remove uninformative of noisy shadow features
    shadows= ['ps_car_11_cat',
              #'ps_calc_14',
              'ps_calc_11',
              'ps_calc_06',
              'ps_calc_16_bin',
              'ps_calc_19_bin',
              'ps_calc_20_bin',
              'ps_calc_15_bin',
              'ps_ind_11_bin',
              'ps_ind_10_bin']
    for s in shadows:
        if s in bin_cols:
            bin_cols.remove(s)
        elif s in num_cols:
            num_cols.remove(s)
        elif s in cat_cols:
            cat_cols.remove(s)
        train_df.drop(s,axis=1,inplace=True)

    for c in train_df.columns:
        if 'calc' in c:
            train_df.drop(c,axis=1,inplace=True)
            if c in bin_cols:
                bin_cols.remove(c)
            elif c in num_cols:
                num_cols.remove(c)
            elif c in cat_cols:
                cat_cols.remove(c)


    train_df= transform_df(train_df)

    #reg_ctr= train_df.groupby('ps_reg_03').target.agg(lambda x: sum(x))
    #train_df.loc[:,'ps_reg_03_bin']= train_df.apply(lambda x: reg_ctr[x['ps_reg_03']],axis=1)
    #test_df.loc[:,'ps_reg_03_bin']= test_df.apply(lambda x: reg_ctr[x['ps_reg_03']] if x['ps_reg_03'] in reg_ctr else 0.,axis=1)

    toPrevalate= ['ps_car_06_cat','ps_car_01_cat','ps_car_11']
    num_cols.remove('ps_car_11')
    toOneHot= filter(lambda x: x not in toPrevalate,cat_cols)

    # one hot encoder
    import sklearn.preprocessing
    enc= sklearn.preprocessing.OneHotEncoder(sparse= False)
    X_train_onehot= enc.fit_transform( train_df.loc[:,toOneHot].as_matrix()+1.0 )

    # Prevalence vectorizer
    vectorizer= prevalence_vectorizer(toPrevalate,target_col)
    vectorizer.train(train_df.loc[:,toPrevalate + [target_col] ])
    X_train_prevalence= np.zeros([train_df[target_col].count(),len(toPrevalate)])
    i=0
    for c in toPrevalate:
        X_train_prevalence[:,i]= vectorizer.parse( train_df.loc[:,c].values, c )
        i += 1

    # Get means and stds for the columns in training set (for non-missing values)
    numerical_norms= {}
    for c in num_cols:
        numerical_norms[c]= (train_df.loc[ train_df[c] >= 0, c].mean(),train_df.loc[ train_df[c] >= 0, c].std())

    # Normalise numerical fields <- TODO: better treatment of [ps_car_14 and ps_reg_03]
    X_train_numeric= np.zeros([train_df[target_col].count(),len(num_cols)])
    i=0
    for c in num_cols:
        X_train_numeric[:,i]= ( train_df[c].values - numerical_norms[c][0] ) / numerical_norms[c][1]
        i+=1

    X_train= np.concatenate([ X_train_numeric,
                             X_train_prevalence,
                             X_train_onehot,
                             train_df.loc[:,bin_cols].as_matrix()
                             ],
                            axis=1)

    X_train[X_train == -1]= np.NaN
    y_train= train_df[target_col].values

    # NOW PREP TEST

    # Remove uninformative of noisy shadow features
    shadows= ['ps_car_11_cat',
#              'ps_calc_14',
              'ps_calc_11',
              'ps_calc_06',
              'ps_calc_16_bin',
              'ps_calc_19_bin',
              'ps_calc_20_bin',
              'ps_calc_15_bin',
              'ps_ind_11_bin',
              'ps_ind_10_bin']
    for s in shadows:
        if s in bin_cols:
            bin_cols.remove(s)
        elif s in num_cols:
            num_cols.remove(s)
        elif s in cat_cols:
            cat_cols.remove(s)
        test_df.drop(s,axis=1,inplace=True)

    for c in test_df.columns:
        if 'calc' in c:
            test_df.drop(c,axis=1,inplace=True)
            if c in bin_cols:
                bin_cols.remove(c)
            elif c in num_cols:
                num_cols.remove(c)
            elif c in cat_cols:
                cat_cols.remove(c)
    test_df= transform_df(test_df)
    X_test_onehot= enc.transform( test_df.loc[:,toOneHot].as_matrix()+1.0 )
    X_test_prevalence= np.zeros([len(test_df.index),len(toPrevalate)])
    i=0
    for c in toPrevalate:
        X_test_prevalence[:,i]= vectorizer.parse( test_df.loc[:,c].values, c )
        i += 1
    X_test_numeric= np.zeros([len(test_df.index),len(num_cols)])
    i=0
    for c in num_cols:
        X_test_numeric[:,i]= ( test_df[c].values - numerical_norms[c][0] ) / numerical_norms[c][1]
        i+=1
    X_test= np.concatenate([ X_test_numeric,
                            X_test_prevalence,
                            X_test_onehot,
                            test_df.loc[:,bin_cols].as_matrix()
                            ],
                           axis=1)
    X_test[X_test == -1]= np.NaN
    if target_col in test_df.columns:
        y_test= test_df[target_col].values
    else:
        y_test= None

    return X_train, y_train, X_test, y_test


def test():
    gc.enable()
    
    trn_df = pd.read_csv("../data/train.csv", index_col=0)
    sub_df = pd.read_csv("../data/test.csv", index_col=0)
    return run_pipe1(trn_df,sub_df)

if __name__ == '__main__':
    test()
