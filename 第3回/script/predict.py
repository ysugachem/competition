# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 01:54:50 2018

@author: Yuki
"""

import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from scipy.stats import rankdata
import gc


# 前処理したデータをロード
ts = pd.read_csv('../input/test_3comb_nmf.csv').sort_values(by='test_id').reset_index(drop=True)
X_cols = ts.columns.drop(['date_time', 'test_id', 'is_arrested']).tolist()

n_ = 5
preds = np.zeros([len(ts), n_])

for i in range(n_):
    gc.collect()
    d_sample = pd.read_csv('../input/under_sampled_{}_lda_nmf.csv'.format(101+i))# baggingのため、複数回モデルを作成
#    clf = xgb.XGBClassifier(eta=0.05, min_child_weight=1, subsample=0.9, colsample_bylevel = 0.2, reg_lambda=1, reg_alpha=0.4)
    clf = LGBMClassifier()
    clf.fit(d_sample[X_cols], d_sample['is_arrested'])    
    preds[:, i] = rankdata(clf.predict_proba(ts[X_cols])[:, 1]) / len(ts)

res = pd.DataFrame()
res['test_id'] = ts['test_id']
res['pred_proba'] = preds.mean(axis=1)
res = res.sort_values(by='test_id')
res.to_csv('../output/sub.csv')
