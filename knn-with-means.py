#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:01:15 2018

@author: cariello
"""

import surprise

import numpy as np
import pandas as pd

work="~/desenv/rcm-portal/dados"
input_csv = r"%s/id_menu+id_usuario+mean_score.csv" % work

df = pd.read_csv(input_csv, delimiter=";")

lower_rating = df['score'].min()
upper_rating = df['score'].max()

print('Score range: {0} to {1}'.format(lower_rating, upper_rating))

reader = surprise.Reader(rating_scale= (.1, upper_rating))
data = surprise.Dataset.load_from_df(df, reader)

alg = surprise.KNNBaseline(k=35, sim_options={'name': 'pearson_baseline', 'user_based': True})
output = alg.fit(data.build_full_trainset())


uids = df['userId'].unique()
#All item ids
iids = df['itemId'].unique()

df_top_rated = pd.DataFrame(columns=['userId', 'itemId', 'est'])

for uid in uids:
    #iids of uid
    iid_of_uid = df.loc[df['userId'] == uid, 'itemId']
    
    #unrated items
    iids_to_pred = np.setdiff1d(iids, iid_of_uid)

    #now, predict the items uid didnt rate and find the top 10
    testset = [ [32,iid, upper_rating] for iid in iids_to_pred]
    predictions = alg.test(testset)
    predictions[0]

    pred_ratings = np.array([pred.est for pred in predictions])

    ind = np.argpartition(pred_ratings, -10)[-10:]
    for i in ind:
        df_top_rated.loc[len(df_top_rated)] = [uid, i, pred_ratings[i]]

output = surprise.model_selection.cross_validate(alg, data, verbose = True)