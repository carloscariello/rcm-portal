#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:01:15 2018

@author: cariello
"""

import surprise

import pandas as pd
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold

work="~/desenv/rcm-portal/dados"
input_csv = r"%s/id_menu+id_usuario+mean_score.csv" % work

df = pd.read_csv(input_csv, delimiter=";")

lower_rating = df['score'].min()
upper_rating = df['score'].max()

reader = surprise.Reader(rating_scale= (.1, upper_rating))
data = surprise.Dataset.load_from_df(df, reader)

kf = KFold(random_state=0)  # folds will be the same for all algorithms.

for k_clusters in range(1,35):
    print("For k={0}".format(k_clusters))
    alg = surprise.KNNBaseline(k=k_clusters, sim_options={'name': 'pearson_baseline', 'user_based': True})
    output = alg.fit(data.build_full_trainset())
    
    out = cross_validate(surprise.KNNBaseline, data, ['rmse', 'mae'], kf)
    
    validate = surprise.model_selection.cross_validate(alg, data, verbose = True)