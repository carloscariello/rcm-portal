#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:19:19 2018

@author: cariello
"""



import pandas as pd

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

work="~/desenv/rcm-portal/dados"
df = None

df = pd.read_csv(r"%s/id_menu+usuario+score.csv" % work, delimiter=";")

df.shape

df.head(20)

max_score = df['score'].max()

print(max_score)

reader = Reader(rating_scale=(0, max_score + 1))

data = Dataset.load_from_df(df[['id_usuario', 'id_menu', 'score']], reader)

trainset, testset = train_test_split(data, test_size=.2)

algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})

algo.fit(trainset)

uid = str(1106)  # raw user id
iid = str(128)  # raw item id

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

test_pred = algo.test(testset)


# get RMSE
print("User-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)
