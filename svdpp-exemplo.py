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

reader = surprise.Reader(rating_scale= (lower_rating, upper_rating))
data = surprise.Dataset.load_from_df(df, reader)

alg = surprise.SVDpp()
output = alg.fit(data.build_full_trainset())

pred = alg.predict(uid='32', iid='649')
score = pred.est
print(score)

#All item ids
iids = df['itemId'].unique()

#iids of uid=32
iid32 = df.loc[df['userId'] == 32, 'itemId']

iids_to_pred = np.setdiff1d(iids, iid32)

#now, predict the items userId 32 didnt rate and find the top 10
testset = [ [32,iid, upper_rating] for iid in iids_to_pred]
predictions = alg.test(testset)
predictions[0]

pred_ratings = np.array([pred.est for pred in predictions])

i_max = pred_ratings.argmax()

iid = iids_to_pred[i_max]
print('Top item for user 50 has iid {0} with predicted rating {1}'.format(iid, pred_ratings[i_max] ))

ind = np.argpartition(pred_ratings, -10)[-10:]

for i in ind:
    print('Estimative rating for item {0} is {1}'.format(i, pred_ratings[i]))