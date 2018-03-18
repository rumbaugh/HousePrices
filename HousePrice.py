import numpy as np
import pandas as pd
import time
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, Lasso
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle

pd.options.mode.chained_assignment = None

def shuffle(x):
    rinds = np.arange(len(x))
    np.random.shuffle(rinds)
    return x[rinds]

def FloatEncode(df, col, encode_nulls = 0, inplace = False):
    if encode_nulls: df[col].fillna(0, inplace = True)
    colvals = df[col].unique()
    for x, i in zip(colvals, range(1, colvals.shape[0] + 1)):
        df[col][df[col] == x] = i
    if not(inplace): 
        return df[col].astype('float')
    else:
        df[col] = df[col].astype('float')


def OneHotEncode_all(df, DropFirst = False):
    df.MSSubClass = df.MSSubClass.astype('string')
    cdict = df.columns.to_series().groupby(df.dtypes).groups
    needEncoding = cdict[dtype('O')]
    for OneHots in needEncoding:
        df = pd.merge(df.drop([OneHots], axis=1), pd.get_dummies(df[OneHots], prefix = OneHots, drop_first=DropFirst), left_index = True, right_index = True)
    return df

class HP():
    def __init__(self, datadir = '.', params = {'n_estimators': 10, 'min_samples_leaf': 2}, internaltestfrac = None, train = None, test = None):
        if np.shape(train) != ():
            self.train = train
        else:
            self.train = pd.read_csv('{}/train.csv'.format(datadir))
        if np.shape(test) != ():
            self.test = test
        else:
            self.test = pd.read_csv('{}/test.csv'.format(datadir))
        self.params = params
        if internaltestfrac != None:
            randinds = shuffle(np.arange(0, self.train.shape[0]))
            self.train, self.test = self.train.iloc[randinds[:int(internaltestfrac*self.train.shape[0])]],  self.train.iloc[randinds[int(internaltestfrac*self.train.shape[0]):]],
        self.orig_train  = self.train.copy()
        self.fill_all_nas(self.orig_train)

    def fill_all_nas(self, df):
        for col in ['LotFrontage', 'MasVnrArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']:
            df[col].fillna(0, inplace = True)
        for col in ['MasVnrType']:
            df[col].fillna('None', inplace = True)
        for col in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','Fence','MiscFeature', 'Alley']:
            df[col].fillna('NA', inplace = True)
        for col in ['Electrical','MSZoning', 'KitchenQual', 'Functional', 'SaleType']:
            df[col].fillna(df[col].value_counts().index[0], inplace = True)
        df.GarageYrBlt[pd.isnull(df.GarageYrBlt)] = df[pd.isnull(df.GarageYrBlt)].YearBuilt
        df.drop(['PoolQC', 'Utilities'], axis = 1, inplace = True)

    def OneHotEncode_DoubleCols(self, df):
        df.Exterior1st[np.in1d(df.Exterior1st, ['Brk Cmn', 'BrkComm', 'BrkFace'])] = 'Brick'
        df.Exterior2nd[np.in1d(df.Exterior2nd, ['Brk Cmn', 'BrkComm', 'BrkFace'])] = 'Brick'
        df.Exterior1st[df.Exterior1st == 'CmentBd'] = 'CemntBd'
        df.Exterior2nd[df.Exterior2nd == 'CmentBd'] = 'CemntBd'
        df.Exterior1st[df.Exterior1st == 'Wd Shng'] = 'WdShing'
        df.Exterior2nd[df.Exterior2nd == 'Wd Shng'] = 'WdShing'
        df.Exterior1st[np.in1d(df.Exterior1st, ['Cblock','AsphShn','Stone','ImStucco'])] = 'Other'
        df.Exterior2nd[np.in1d(df.Exterior2nd, ['Cblock','AsphShn','Stone','ImStucco'])] = 'Other'
        df.Condition1[np.in1d(df.Condition1, ['RRAe','RRAn','RRNe','RRNn'])] = 'RR'
        df.Condition2[np.in1d(df.Condition2, ['RRAe','RRAn','RRNe','RRNn'])] = 'RR'
        df.Condition1[np.in1d(df.Condition1, ['PosA','PosN'])] = 'Pos'
        df.Condition2[np.in1d(df.Condition2, ['PosA','PosN'])] = 'Pos'
        for col1, col2, base in zip(['Exterior1st', 'Condition1'], ['Exterior2nd', 'Condition2'], ['Exterior', 'Condition']):
            for colval in df[col1].value_counts().add(df[col2].value_counts(), fill_value = 0).index.values:
                newcol = '{}_{}'.format(base, colval)
                df[newcol] = 0
                df[newcol][(df[col1] == colval) | (df[col2] == colval)] = 1
            df.drop([col1, col2], axis = 1, inplace = True)
        df['BsmtFinSF'] = df['BsmtFinSF1'].add(df['BsmtFinSF2'], fill_value = 0)
        df['BsmtFinMixed'] = 0
        df['BsmtFinMixed'][df['BsmtFinType2'] != 'NA'] = 1
        df['PorchSF'] = df['OpenPorchSF'].add(df['EnclosedPorch'], fill_value = 0).add(df['3SsnPorch'], fill_value = 0).add(df['ScreenPorch'], fill_value = 0)
        df.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis = 1, inplace = True)

    def refine_df(self, df):
        df['MSSubClass']['MSSubClass' < 90] = 1 #Normal house
        df['MSSubClass']['MSSubClass' >180] = 4 # 2 Family Conversion
        df['MSSubClass']['MSSubClass' >100] = 3 # Planned Unit Development
        df['MSSubClass']['MSSubClass' >85] = 2 # Duplex


    def setup_df(self, df, dropdoubles = False, OneHotEncoding = False, addsqrtarea = False, logtransform = False):
        self.fill_all_nas(df)
        if dropdoubles:
            df['BsmtFinSF'] = df['BsmtFinSF1'].add(df['BsmtFinSF2'], fill_value = 0)
            df.drop(['Exterior1st','Exterior2nd','Condition1','Condition2','BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis = 1, inplace = True)
        else:
            self.OneHotEncode_DoubleCols(df)
        if addsqrtarea:
            for col in ['BsmtFinSF', 'LotArea', '2ndFlrSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']:
                df['{}_sqrt'.format(col)] = np.sqrt(df[col].values)
        cdict = df.columns.to_series().groupby(df.dtypes).groups
        if logtransform:
            for col in ['LotArea','BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'BsmtFinSF','LotFrontage', 'MasVnrArea','PorchSF','SalePrice']:
                try:
                    df[col] = np.log1p(df[col].values)
                except:
                    print 'Failed logtransform on {}'.format(col)
        if OneHotEncoding:
            df = OneHotEncode_all(df)
        else:
            for col in cdict[dtype('O')]: FloatEncode(df, col, inplace = True)
        df.drop(['Id'], axis = 1, inplace = True)
        return df


    def setup_df_all(self, dropdoubles = False, OneHotEncoding = False, addsqrtarea = False, logtransform = False, pwrterms = None):
        all_data = pd.concat((self.train.loc[:,'MSSubClass':'SaleCondition'],
                      self.test.loc[:,'MSSubClass':'SaleCondition']))
        self.fill_all_nas(all_data)
        if np.shape(pwrterms) == ():
            if pwrterms == None:
                pwrterms = []
            else:
                pwrterms = np.array([pwrterms])
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index.values
        for pwr in pwrterms:
            for col in numeric_feats:
                all_data['{}_sqr'.format(col)] = all_data[col].values**pwr
        if logtransform:
            self.train['SalePrice'] = np.log1p(self.train['SalePrice'])
            numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
            skewed_feats = all_data[:self.train.shape[0]][numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
            skewed_feats = skewed_feats[skewed_feats > 0.75]
            skewed_feats = skewed_feats.index
            all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
        if dropdoubles:
            all_data['BsmtFinSF'] = all_data['BsmtFinSF1'].add(all_data['BsmtFinSF2'], fill_value = 0)
            all_data.drop(['Exterior1st','Exterior2nd','Condition1','Condition2','BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis = 1, inplace = True)
        else:
            self.OneHotEncode_DoubleCols(all_data)
        if addsqrtarea:
            for col in ['BsmtFinSF', 'LotArea', '2nall_datalrSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea']:
                all_data['{}_sqrt'.format(col)] = np.sqrt(all_data[col].values)
        cdict = all_data.columns.to_series().groupby(all_data.dtypes).groups
        if OneHotEncoding:
            all_data = pd.get_dummies(all_data)
        else:
            for col in cdict[dtype('O')]: FloatEncode(all_data, col, inplace = True)
        self.X_train = all_data[:self.train.shape[0]]
        self.X_test = all_data[self.train.shape[0]:]
        self.y = self.train.SalePrice
        

    def xgb_fit(self, cv_params = {'max_depth': 2, 'eta': 0.1}, params = {'n_estimators': 1000, 'max_depoth': 2, 'learning_rate': 0.1}, OneHotEncoding = True, addsqrtarea = False, dropdoubles = False, logtransform = True, pwrterms = None):
        if np.shape(params) == (): params = self.params
        self.setup_df_all(OneHotEncoding = OneHotEncoding, addsqrtarea = addsqrtarea, dropdoubles = dropdoubles, logtransform = logtransform, pwrterms = pwrterms)
        self.dtrain = xgb.DMatrix(self.X_train, label = self.y)
        self.dtest = xgb.DMatrix(self.X_test)
        self.reg_cv = xgb.cv(cv_params, self.dtrain, num_boost_round = 500, early_stopping_rounds = 100)
        self.reg = xgb.XGBRegressor(**params)
        self.reg.fit(self.X_train, self.y)
        preds = self.reg.predict(self.X_test)
        return preds

    def fit(self, params = None, reg = RandomForestRegressor, OneHotEncoding = True, addsqrtarea = False, dropdoubles = False, logtransform = True, pwrterms = None):
        if np.shape(params) == (): params = self.params
        self.setup_df_all(OneHotEncoding = OneHotEncoding, addsqrtarea = addsqrtarea, dropdoubles = dropdoubles, logtransform = logtransform, pwrterms = pwrterms)
        self.reg = reg(**params)
        self.reg.fit(self.X_train, self.y)
        preds = self.reg.predict(self.X_test)
        return preds

    def simple_fit(self, params = None, reg = RandomForestRegressor, OneHotEncoding = False, addsqrtarea = False, dropdoubles = False, logtransform = False):
        if np.shape(params) == (): params = self.params
        self.train = self.setup_df(self.train, OneHotEncoding = OneHotEncoding, addsqrtarea = addsqrtarea, dropdoubles = dropdoubles, logtransform = logtransform)
        self.test = self.setup_df(self.test, OneHotEncoding = OneHotEncoding, addsqrtarea = addsqrtarea, dropdoubles = dropdoubles, logtransform = logtransform)
        #if OneHotEncoding:
        for col in np.array(self.train.columns.tolist())[np.in1d(self.train.columns.tolist(), self.test.columns.tolist(), invert = True)]:
            if col != 'SalePrice': self.test[col] = 0
        for col in np.array(self.test.columns.tolist())[np.in1d(self.test.columns.tolist(), self.train.columns.tolist(), invert = True)]:
            self.train[col] = 0
        self.reg = reg(**params)
        self.reg.fit(self.train.drop(['SalePrice'], axis = 1), self.train.SalePrice)
        preds = self.reg.predict(self.test.loc[:, self.test.columns != 'SalePrice'])
        return preds

    def two_stage_fit(self, second_stage_cols = ['SaleType', 'SaleCondition'], trainfrac = 1, params = None, reg = RandomForestRegressor, OneHotEncoding = False, addsqrtarea = False, dropdoubles = False, logtransform = False):
        if np.shape(params) == (): params = self.params
        self.train = self.setup_df(self.train, OneHotEncoding = OneHotEncoding, addsqrtarea = addsqrtarea, dropdoubles = dropdoubles, logtransform = logtransform)
        self.test = self.setup_df(self.test, OneHotEncoding = OneHotEncoding, addsqrtarea = addsqrtarea, dropdoubles = dropdoubles, logtransform = logtransform)
        #if OneHotEncoding:
        for col in np.array(self.train.columns.tolist())[np.in1d(self.train.columns.tolist(), self.test.columns.tolist(), invert = True)]:
            if col != 'SalePrice': self.test[col] = 0
        for col in np.array(self.test.columns.tolist())[np.in1d(self.test.columns.tolist(), self.train.columns.tolist(), invert = True)]:
            self.train[col] = 0
        self.reg = reg(**params)            
        if trainfrac < 1:
            randinds = shuffle(np.arange(0, self.train.shape[0]))
            X, X2 = self.train.iloc[:int(trainfrac*self.train.shape[0])].drop(second_stage_cols, axis = 1), self.train.iloc[int(trainfrac*self.train.shape[0]):]
        else:
            X = self.train.drop(second_stage_cols, axis = 1)
            X2 = X.copy()
        self.reg.fit(X.drop(['SalePrice'], axis = 1), X.SalePrice)
        preds_prelim = self.reg.predict(X2.drop(['SalePrice'] + second_stage_cols, axis = 1))
        self.reg2 = reg(**params)
        X2 = X2[['SalePrice'] + second_stage_cols]
        X2['preliminary_prediction'] = preds_prelim
        self.reg2.fit(X2.drop(['SalePrice'], axis = 1), X2.SalePrice)
        preds0 = self.reg.predict(self.test.drop(second_stage_cols, axis = 1))
        Xtest = self.test[second_stage_cols]
        Xtest['preliminary_prediction'] = preds0
        preds = self.reg2.predict(Xtest)
        return preds

    def qual_split_fit(self, params = None, reg = RandomForestRegressor, OneHotEncoding = False, qualthreshes = [0, 4, 5, 6, 7, 10]):
        if np.shape(params) == (): params = self.params
        self.train = self.setup_df(self.train, OneHotEncoding = OneHotEncoding)
        self.test = self.setup_df(self.test, OneHotEncoding = OneHotEncoding)
        for qual_lo, qual_hi, iq in zip(qualthreshes[:-1], qualthreshes[1:], range(0,len(qualthreshes) - 1)):
            self.train.OverallQual[(self.train.OverallQual > qual_lo) & (self.train.OverallQual <= qual_hi)] = iq
            self.test.OverallQual[(self.test.OverallQual > qual_lo) & (self.test.OverallQual <= qual_hi)] = iq
        #if OneHotEncoding:
        for col in np.array(self.train.columns.tolist())[np.in1d(self.train.columns.tolist(), self.test.columns.tolist(), invert = True)]:
            if col != 'SalePrice': self.test[col] = 0
        for col in np.array(self.test.columns.tolist())[np.in1d(self.test.columns.tolist(), self.train.columns.tolist(), invert = True)]:
            self.train[col] = 0
        self.regs, preds = np.zeros(len(qualthreshes) - 1, dtype = 'object'), np.zeros(self.test.shape[0])
        for iq in range(0,len(qualthreshes) - 1):
            self.regs[iq] = reg(**params)
            X, testmask = self.train[self.train.OverallQual == iq], (self.test.OverallQual == iq)
            print X.shape, testmask.shape
            self.regs[iq].fit(X.drop(['SalePrice'], axis = 1), X.SalePrice)
            preds[testmask] = self.regs[iq].predict(self.test[testmask].drop(['SalePrice'], axis = 1))
        return preds

    def set_training_preds(self):
        self.preds_train = self.reg.predict(self.train.drop(['SalePrice'], axis = 1))
        self.resids = (self.preds_train - self.train['SalePrice']) / self.train['SalePrice']

    def resid_comp(self, col, figure = 1, clear = True, colors = ['red', 'cyan', 'blue', 'silver', 'green', 'orange','#C0E200', 'pink','gold','magenta','brown','purple','#D2CCF8','darkblue','yellow','#EFF8CC','#44FBBB'], markersize = 6, force_value_counts = False):
        plt.figure(figure)
        if clear: plt.clf()
        #plt.scatter(self.orig_train['SalePrice'], self.resids, label = 'All', color = colors[0])
        if (self.orig_train[col].dtype == dtype('O')) | (force_value_counts):
            while len(colors) < len(self.orig_train[col].value_counts().index):
                colors = np.append(colors, colors)
            for colval, i_col in zip(self.orig_train[col].value_counts().index, range(0, len(self.orig_train[col].value_counts().index))):
                plt.scatter(self.orig_train['SalePrice'][self.orig_train[col] == colval], self.resids[self.orig_train[col] == colval], label = colval, color = colors[i_col], s = markersize)
        else:
            description = self.orig_train[col].describe(percentiles = [.125, .375, .625, .875])
            threshes = ['min', '12.5%', '37.5%', '62.5%', '87.5%', 'max']
            for lo_ind, hi_ind, i_col in zip(threshes[:-1], threshes[1:], range(1, len(threshes))):
                thresh_mask = (self.orig_train[col] >= description.loc[lo_ind]) & (self.orig_train[col] <= description.loc[hi_ind])
                plt.scatter(self.orig_train['SalePrice'][thresh_mask], self.resids[thresh_mask], label = '{}-{}'.format(lo_ind, hi_ind), color = colors[i_col-1], s = markersize)
        plt.axhline(0, color = 'k', lw = 2, ls = 'dashed')
        plt.legend(frameon = False)
        plt.ylabel('Sales Price Residuals')
        plt.xlabel('Actual Sales Price')
        plt.show(block = False)

def multi_model(weights = np.array([0.25, 0.5, 0.25])):
    HP1 = HP()
    predsRF = HP1.simple_fit()
    HPls = HP(params = {'alphas': [1, 0.1, 0.001, 0.0005], 'max_iter': 100000})
    preds_ls = HPls.fit(reg = LassoCV, OneHotEncoding=True, logtransform=True)
    HPxgb = HP(params = {})
    preds_xgb = HPxgb.xgb_fit()
    preds = weights[0]*predsRF + weights[1]*np.expm1(preds_ls) + weights[2]*np.expm1(preds_xgb)
    return preds

def stack_model(internaltestfrac = 0.6, pwrterms = None, datadir = '.', final_cv_params = {'max_depth': 2, 'eta': 0.1}, NFOLD = 5):
    HP_test = HP()
    HP_test.setup_df_all(pwrterms = pwrterms, OneHotEncoding=True, logtransform=True)
    test = HP_test.X_test
    train = pd.read_csv('{}/train.csv'.format(datadir))
    Xsize = train.shape[0]
    preds = np.zeros((NFOLD, test.shape[0]))
    randinds = shuffle(np.arange(0, Xsize))
    for nf in range(0,NFOLD):
        HP_testRF = HP()
        HP_testRF.setup_df(HP_testRF.test)
        train_cut, test_cut = train.iloc[np.roll(randinds, (nf*Xsize)/NFOLD)[Xsize/NFOLD:]], train.iloc[np.roll(randinds, (nf*Xsize)/NFOLD)[:Xsize/NFOLD]]
        HP1 = HP(train = train_cut.copy(), test = test_cut.copy())
        predsRF_train = np.log1p(HP1.simple_fit())
        predsRF_test = np.log1p(HP1.reg.predict(HP_testRF.test))
        HPls = HP(train = train_cut.copy(), test = test_cut.copy(), params = {'alphas': [1, 0.1, 0.001, 0.0005], 'max_iter': 100000})
        preds_ls_train = HPls.fit(reg = LassoCV, OneHotEncoding=True, logtransform=True, pwrterms = pwrterms)
        preds_ls_test = HPls.reg.predict(test)
        HPxgb = HP(train = train_cut.copy(), test = test_cut.copy(), params = {})
        preds_xgb_train = HPxgb.xgb_fit(pwrterms = pwrterms, OneHotEncoding=True, logtransform=True)
        preds_xgb_test = HPxgb.reg.predict(test)
        X_secondary_train, X_secondary_test = pd.DataFrame({'predsRF': predsRF_train, 'preds_ls': preds_ls_train, 'preds_xgb': preds_xgb_train}), pd.DataFrame({'predsRF': predsRF_test, 'preds_ls': preds_ls_test, 'preds_xgb': preds_xgb_test})
        y_sec = np.log1p(test_cut.SalePrice)
        dtrain = xgb.DMatrix(X_secondary_train, label = y_sec)
        dtest = xgb.DMatrix(X_secondary_test)
        reg_cv = xgb.cv(final_cv_params, dtrain, num_boost_round = 500, early_stopping_rounds = 100)
        reg = xgb.XGBRegressor()
        reg.fit(X_secondary_train, y_sec)
        preds[nf] = reg.predict(X_secondary_test)
    return preds.mean(axis = 0), reg

def get_oof(reg, x_train, y_train, x_test, NFOLDS = 5):
    ntrain, ntest = x_train.shape[0], x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    shuffle_inds = np.arange(ntrain)
    np.random.shuffle(shuffle_inds)

    for ikf in range(0, NFOLDS):
        test_index, train_index = shuffle_inds[int(ikf*ntrain/NFOLDS):int((ikf+1)*ntrain/NFOLDS)], np.append(shuffle_inds[:int(ikf*ntrain/NFOLDS)], shuffle_inds[int((ikf+1)*ntrain/NFOLDS):])
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        reg.train(x_tr, y_tr)

        oof_train[test_index] = reg.predict(x_te)
        oof_test_skf[ikf, :] = reg.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


#Notes - Overall Quality correlates highly with price. I could carry out regression separately for different tiers (this made it worse)
#
# Where Exterior condition is Fair (basically the worst houses), there seem to be relatively high residuals. Maybe sort condition into bad and good, where good is everything TA and better. Seems to be true for other conditions, like Basement
# Gas also seems to have high residuals. Maybe sort into GasA and Other, because it's almost all GasA
# High residuals for no central air
#LowQualFinSF  has like all zeros except for 20 outliers. I should maybe drop it or consolidate all non-zero entries into one value
#PavedDrive P and N should maybe be consolidated as N
# Too many porch types. Maybe just add all the porch areas together
# For SaleType, New houses seem to have higher residuals than others. Maybe break into down into just New and not
# Some of the highest residuals seem to be not Normal SaleConditions. Some things I could do: break it up into Normal, Partial, and Other; keep this out of initial fit, then do a fit using this feature and the predictions with a secondary training set.
