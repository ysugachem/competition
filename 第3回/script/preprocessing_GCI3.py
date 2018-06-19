# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 22:03:33 2018

@author: Yuki
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold

"""
データのimport 及び結合

予測提出の際にtest_id順に並べる必要があるのでtest_idも持っておく

"""
df_tr = pd.read_csv('../input/train.csv')
df_ts = pd.read_csv('../input/test.csv')
df_ts['test_id'] = df_ts.index
df_tr['test_id'] = -1
df_all = pd.concat([df_tr, df_ts]).sort_values(by=['stop_date', 'stop_time'])
df_all['officer_id_original'] = df_all['officer_id'].values
del df_all['county_name'], df_all['driver_race'], df_all['search_type'], df_all['state'], df_all['driver_age'], df_all['police_department'], df_all['violation'], df_all['search_conducted'], df_all['fine_grained_location']


"""
時間不明のやつはflagは立てる
サンプルは捨てたくないから、他のサンプルの出現回数に応じた重みで時間つける
"""
import random
random.seed(71)
time_completion = random.choices(df_all['stop_time'].value_counts().index.tolist(), k=df_all['stop_time'].isna().sum(), weights=df_all['stop_time'].value_counts().values.tolist())
df_all['time_na'] = 0
df_all.loc[df_all['stop_time'].isna(), 'time_na'] = 1
df_all.loc[df_all['stop_time'].isna(), 'stop_time'] = time_completion
df_all['stop_time'] = df_all['stop_time'].fillna(df_all['stop_time'].value_counts().index[0])
df_all['date_time'] = df_all['stop_date'] + ' ' + df_all['stop_time']
del df_all['stop_time'], df_all['stop_date']
df_all.index = pd.DatetimeIndex(df_all['date_time'])
df_all = df_all.sort_index()
print(df_all.nunique())

"""
コラムの名前
"""
# columnsの名前をグループ分けしておく
cols_cat_u1000 = ['contraband_found', 'county_fips', 'driver_gender', 'driver_race_raw', 'location_raw', 'search_type_raw', 'stop_duration', 'violation_raw', 'time_na']
cols_cat_1000 = ['officer_id']
cols_cat_u1000 = cols_cat_u1000 + cols_cat_1000　#水準1000以下と分けてたけど、結局まとめた
# cols_cat_many = ['fine_grained_location'] 多すぎて使うの大変なので後回し
cols_time = ['date_time']
cols_continuous = ['driver_age_raw']
cols_support = ['test_id']
col_target = 'is_arrested'
df_all.nunique()
# 見づらいので並び替え
#df_all = df_all[cols_cat_u1000+cols_cat_1000+cols_continuous+cols_time+cols_support+[col_target]]
df_all = df_all[cols_cat_u1000+cols_continuous+cols_time+cols_support+[col_target]]


"""
ラベルエンコーディング
"""
X_cols = df_all.columns.drop(['test_id', 'is_arrested'])
# categoryをlabel encoding　水準数が1000以下のもの
for col in cols_cat_u1000:
    df_all[col] = pd.factorize(df_all[col])[0]
for col in cols_cat_1000:
    df_all[col] = pd.factorize(df_all[col])[0]


"""
日付や曜日
"""

df_all['DoW'] = df_all.index.weekday
cols_cat_u1000 = cols_cat_u1000 + ['DoW']
df_all['day'] = pd.to_datetime(df_all['date_time']).dt.day
df_all['month'] = pd.to_datetime(df_all['date_time']).dt.month
df_all['day_of_year'] = (pd.to_datetime(df_all['date_time']) - pd.datetime(2012, 12, 31)).dt.days % 365
df_all['hour'] = pd.to_datetime(df_all['date_time']).dt.hour
df_all['hour_sin'] = np.sin(df_all['hour'] / 24)
df_all['hour_cos'] = np.cos(df_all['hour'] / 24)
df_all['day_sin'] = np.sin(df_all['day_of_year'] / 365)
df_all['day_cos'] = np.cos(df_all['day_of_year'] / 365)

"""
多めのカテゴリカル変数
これらを主にfeature engineering
"""
cols = ['officer_id', 'violation_raw', 'location_raw', 'day_of_year', 'county_fips']


"""
count_unique
"""
def count_unique(df_, cols):
    df = df_.copy()
    new_cols = []
    for col1 in cols:
        for col2 in cols:
            if col1 != col2:
                col2_by_col1 = df.groupby(col1)[col2].nunique()
                df[col2+'_by_'+col1+'_count'] = df[col1].map(col2_by_col1)
                new_cols.append(col2+'_by_'+col1+'_count')
    return df[new_cols]
res = count_unique(df_all, cols)

df_all = pd.concat([df_all, res], axis=1)

"""
トピックモデルの為に、
"""

from sklearn.decomposition import LatentDirichletAllocation, NMF
def nmf_(df_, cols, n_topic=5):
    df = df_.copy()
    nmf = NMF(n_components=n_topic)
    new_cols = []
    for col1 in cols:
        for col2 in cols:
            if col1 != col2:
                a = df.pivot_table(values=['contraband_found'], columns = [col1], index = [col2], aggfunc = 'count', fill_value = 0)
                res = nmf.fit_transform(a)
                res2 = df.groupby(col2)[col1].count()
                for i in range(n_topic):
                    df[col2+'_by_'+col1+'_topic'+str(i+1)+'_NMF'] = df[col2].map(pd.Series(res[:, i], index=res2.index))
                    new_cols.append(col2+'_by_'+col1+'_topic'+str(i+1)+'_NMF')
    return df[new_cols]
def lda(df_, cols, n_topic=5):
    df = df_.copy()
    LDA = LatentDirichletAllocation(n_components=n_topic)
    new_cols = []
    for col1 in cols:
        for col2 in cols:
            if col1 != col2:
                a = df.pivot_table(values=['contraband_found'], columns = [col1], index = [col2], aggfunc = 'count', fill_value = 0)
                res = LDA.fit_transform(a)
                res2 = df.groupby(col2)[col1].count()
                for i in range(n_topic):
                    df[col2+'_by_'+col1+'_topic'+str(i+1)] = df[col2].map(pd.Series(res[:, i], index=res2.index))
                    new_cols.append(col2+'_by_'+col1+'_topic'+str(i+1))
    return df[new_cols]
res = lda(df_all, cols)
df_all = pd.concat([df_all, res], axis=1)
res = nmf_(df_all, cols)
df_all = pd.concat([df_all, res], axis=1)


"""
next click
"""
# next系の実装
# 組み合わせ変数に対してもやる必要ある？
df_all['time_int'] = np.array(df_all.index - pd.datetime(2013, 10, 1), dtype=int) / 60000000000

def next_click(df_, cols):
    df = df_.copy()
    new_cols = []
    for col in cols:
        df[col+'_time_interval_before'] = np.nan
        df[col+'_time_interval_after'] = np.nan
        new_cols.append(col+'_time_interval_before')
        new_cols.append(col+'_time_interval_after')
        
        unique_values = df[col].unique()
        for i in unique_values:
            a = df.loc[df[col]==i, 'time_int'].values
            time_interval_before = np.zeros(len(a))
            time_interval_after = np.zeros(len(a))
            time_interval_before[1:] = a[1:] - a[:-1]
            time_interval_after[:-1] = a[:-1] - a[1:]
            time_interval_before[0] = np.median(time_interval_before[1:])
            time_interval_after[-1] = np.median(time_interval_after[:-1])
            df.loc[df[col]==i, col+'_time_interval_before'] = time_interval_before
            df.loc[df[col]==i, col+'_time_interval_after'] = time_interval_after


        df[[col+'_time_interval_after']] = df[[col+'_time_interval_after']].fillna(df[[col+'_time_interval_after']].min()-1)
        df[[col+'_time_interval_before']] = df[[col+'_time_interval_before']].fillna(df[[col+'_time_interval_before']].max()+1)
    return df[new_cols]
df_all['time_int'] = np.array(df_all.index - pd.datetime(2013, 10, 1), dtype=int) / 60000000000
res = next_click(df_all, cols[:3])
df_all = pd.concat([df_all, res], axis=1)


"""
時系列的な流れの特徴
"""
def data_count_time_posterity(df_, post_time='3600s'):
    df = df_.copy()
    df_reverse = df[::-1]
    df_reverse.index = pd.datetime(2050, 1, 1) - df_reverse.index
    res = pd.DataFrame(df_reverse['contraband_found'].rolling(post_time).count()).fillna(1)
    res.columns=['datacount_{}_post'.format(post_time)]
    return res
def data_ysum_time_posterity(df_, post_time='3600s'):
    df = df_.copy()
    df_reverse = df[::-1]
    df_reverse.index = pd.datetime(2050, 1, 1) - df_reverse.index
    res = pd.DataFrame(df_reverse['is_arrested'].rolling(post_time).sum()).fillna(0)
    res.columns=['ysum_{}_post'.format(post_time)]
    res['is_arrested'] = df_reverse['is_arrested']
    res.loc[res.is_arrested==1, 'ysum_{}_post'.format(post_time)] -= 1
    return res[['ysum_{}_post'.format(post_time)]]

def time_series_feature_posterity(df_, post_time='1H'):
    df = df_.copy()
    a = data_count_time_posterity(df, post_time)
    b = data_ysum_time_posterity(df, post_time)
    c = pd.concat([a, b], axis=1)
    c['mean_{}_post'.format(post_time)] = c['ysum_{}_post'.format(post_time)] / c['datacount_{}_post'.format(post_time)]
    new_cols = ['mean_{}_post'.format(post_time), 'ysum_{}_post'.format(post_time), 'datacount_{}_post'.format(post_time)]
    df['mean_{}_post'.format(post_time)] = c['mean_{}_post'.format(post_time)].values[::-1]
    df['ysum_{}_post'.format(post_time)] = c['ysum_{}_post'.format(post_time)].values[::-1]
    df['datacount_{}_post'.format(post_time)] = c['datacount_{}_post'.format(post_time)].values[::-1]
    return df[new_cols].fillna(0)

def data_count_time_priority(df_, prior_time='3600s'):
    df = df_.copy()
    res = pd.DataFrame(df['contraband_found'].rolling(prior_time).count()).fillna(1)
    res.columns=['datacount_{}_prior'.format(prior_time)]
    return res
def data_ysum_time_priority(df_, prior_time='3600s'):
    df = df_.copy()
    res = pd.DataFrame(df['is_arrested'].rolling(prior_time).sum()).fillna(0)
    res.columns=['ysum_{}_prior'.format(prior_time)]
    res['is_arrested'] = df['is_arrested']
    res.loc[res.is_arrested==1, 'ysum_{}_prior'.format(prior_time)] -= 1
    return res[['ysum_{}_prior'.format(prior_time)]]
def time_series_feature(df_, prior_time='1H'):
    df = df_.copy()
    a = data_count_time_priority(df, prior_time)
    b = data_ysum_time_priority(df, prior_time)
    c = pd.concat([a, b], axis=1)
    c['mean_{}_prior'.format(prior_time)] = c['ysum_{}_prior'.format(prior_time)] / c['datacount_{}_prior'.format(prior_time)]
    return c.fillna(0)
def time_series_feature_pripos(df_, time_span='600s'):
    df = df_.copy()
    df_pos = time_series_feature_posterity(df, time_span)
    df_pri = time_series_feature(df, time_span)
    res_concat = pd.concat([df_pos, df_pri], axis=1)
    res_concat['datacount_pospri_{}'.format(time_span)] = res_concat['datacount_{}_post'.format(time_span)] + res_concat['datacount_{}_prior'.format(time_span)]
    res_concat['ysum_pospri_{}'.format(time_span)] = res_concat['ysum_{}_post'.format(time_span)] + res_concat['ysum_{}_prior'.format(time_span)]
    res_concat['mean_pospri_{}'.format(time_span)] = res_concat['ysum_pospri_{}'.format(time_span)] / res_concat['datacount_pospri_{}'.format(time_span)]
    return res_concat
df_1H = time_series_feature_pripos(df_all, '1800s')
df_2H = time_series_feature_pripos(df_all, '1H')
df_6H = time_series_feature_pripos(df_all, '3H')
df_12H = time_series_feature_pripos(df_all, '6H')
df_24H = time_series_feature_pripos(df_all, '12H')
df_72H = time_series_feature_pripos(df_all, '36H')
df_168H = time_series_feature_pripos(df_all, '84H')
df_all = pd.concat([df_1H, df_2H, df_6H, df_12H, df_24H, df_72H, df_168H, df_all], axis=1)


"""
既にあるlabelから組み合わせlabelを作成
"""
df_2comb = pd.DataFrame()
for i, col1 in enumerate(cols_cat_u1000):
    for ii, col2 in enumerate(cols_cat_u1000):
        if (i < ii):
            df_2comb[col1+'+'+col2] = df_all[col1].map(str) + '_' + df_all[col2].map(str)
print('2combination')
df_3comb = pd.DataFrame()
for i, col1 in enumerate(cols_cat_u1000):
    for ii, col2 in enumerate(cols_cat_u1000):
        for iii, col3 in enumerate(cols_cat_u1000):
            if (i < ii) & (ii < iii):
                df_3comb[col1+'+'+col2+'+'+col3] = df_all[col1].map(str) + '_' + df_all[col2].map(str) + '_' + df_all[col3].map(str)
print('3combination')

# df_4comb = pd.DataFrame()
# for i, col1 in enumerate(cols_cat_u1000):
#     for ii, col2 in enumerate(cols_cat_u1000):
#         for iii, col3 in enumerate(cols_cat_u1000):
#             for iv, col4 in enumerate(cols_cat_u1000):
#                 if (i < ii) & (ii < iii):
#                     df_4comb[col1+'+'+col2+'+'+col3+'+'+col4] = df_all[col1].map(str) + '_' + df_all[col2].map(str) + '_' + df_all[col3].map(str) + '_' + df_all[col4].map(str)
# print('4combination')

df_comb = pd.concat([df_2comb, df_3comb], axis=1)
#df_comb = df_2comb
for col in df_comb.columns:
    print(col)
    df_comb[col] = pd.factorize(df_comb[col])[0]
cols_comb = df_comb.columns.tolist()
df_all = pd.concat([df_comb, df_all], axis=1)
print('df_shape', df_all.shape)


"""
カテゴリカル変数が全部できた！！
label encodingしたものについて count encoding, mean encoding
"""
def mean_test_encoding(df_trn, df_tst, cols, target):    
    
    for col in cols:
        df_tst[col + '_mean_encoded'] = np.nan
        
    for col in cols:
        tr_mean = df_trn.groupby(col)[target].mean()
        mean = df_tst[col].map(tr_mean)
        df_tst[col + '_mean_encoded'] = mean

    prior = df_trn[target].mean()

    for col in cols:
        df_tst[col + '_mean_encoded'].fillna(prior, inplace = True) 
        
    return df_tst
def mean_train_encoding(df, cols, target):
    y_tr = df[target].values
    skf = StratifiedKFold(5, shuffle = True, random_state=71)

    for col in cols:
        df[col + '_mean_encoded'] = np.nan

    for trn_ind , val_ind in skf.split(df,y_tr):
        x_tr, x_val = df.iloc[trn_ind], df.iloc[val_ind]

        for col in cols:
            tr_mean = x_tr.groupby(col)[target].mean()
            mean = x_val[col].map(tr_mean)
            df[col + '_mean_encoded'].iloc[val_ind] = mean

    prior = df[target].mean()

    for col in cols:
        df[col + '_mean_encoded'].fillna(prior, inplace = True) 
        
    return df
def count_encoding(df_trn, cols, target):
    for col in cols:
        df_trn[col + '_count_encoded'] = np.nan
        
    for col in cols:
        tr_count = df_trn.groupby(col)[target].count()
        df_trn[col + '_count_encoded'] = df_trn[col].map(tr_count)
    return df_trn
cols_cat = cols_cat_u1000+cols_comb
df_all = count_encoding(df_all, cols_cat, col_target)

"""
将来n[s]に何回同じcategory値が出てきたか
"""
def count_encoding_time_posterity(df_, columns, post_time='3600s'):
    df = df_.copy()
    df = df[columns]
    new_cols = []
    for col in columns:
        new_col = col+'_'+post_time+'_count_post'
        df[new_col] = np.nan
        new_cols.append(new_col)
    df_reverse = df[columns][::-1]
    df_reverse.index = pd.datetime(2050, 1, 1) - df_reverse.index
    df[new_cols] = df_reverse.rolling(post_time).apply(lambda x: (x==x[-1]).sum()).values[::-1]
    df = df[new_cols]
    return df

res1800 = count_encoding_time_posterity(df_all, cols_cat, '1800s')
res3600 = count_encoding_time_posterity(df_all, cols_cat, '3600s')
res10800 = count_encoding_time_posterity(df_all, cols_cat, '10800s')
res21600 = count_encoding_time_posterity(df_all, cols_cat, '21600s')
df_all = pd.concat([res21600, res10800, res3600, res1800, df_all], axis=1)
#df_all = pd.concat([res21600, res1800, df_all], axis=1)

def count_encoding_time_priority(df_, columns, prior_time='3600s'):
    df = df_.copy()
    df = df[columns]
    new_cols = []
    for col in columns:
        new_col = col+'_'+prior_time+'_count_prior'
        df[new_col] = np.nan
        new_cols.append(new_col)
    df[new_cols] = df[columns].rolling(prior_time).apply(lambda x: (x==x[-1]).sum())
    df = df[new_cols]
    return df

res1800 = count_encoding_time_priority(df_all, cols_cat, '1800s')
res3600 = count_encoding_time_priority(df_all, cols_cat, '3600s')
res10800 = count_encoding_time_priority(df_all, cols_cat, '10800s')
res21600 = count_encoding_time_priority(df_all, cols_cat, '21600s')
df_all = pd.concat([res21600, res10800, res3600, res1800, df_all], axis=1)


del  df_2comb
del  df_3comb
del  df_comb
del  df_tr
del  df_ts
del  res1800
del  res3600
del  res10800
del res21600
gc.collect()
"""

"""
ts = df_all.loc[df_all.test_id>-0.5]
tr = df_all.loc[df_all.test_id<-0.5]
del df_all
X_cols = tr.columns.drop(['date_time', 'test_id', 'is_arrested']).tolist()
"""

"""
ts = mean_test_encoding(tr, ts, cols_cat, col_target)
tr = mean_train_encoding(tr, cols_cat, col_target)

"""
前処理したデータを保存しておく
"""



ts.to_csv('../input/test_3comb_lda_nmf.csv', index=False)
for i in range(15):
    pos_ = tr.loc[tr.is_arrested==1]
    neg_ = tr.loc[tr.is_arrested==0]
    neg_tmp = neg_.sample(n=len(pos_), random_state=100+i)
    d_sample = pd.concat([pos_, neg_tmp])
    del pos_, neg_, neg_tmp
    gc.collect()
    d_sample.to_csv('../input/under_sampled_{}_lda_nmf.csv'.format(100+i), index=False)






