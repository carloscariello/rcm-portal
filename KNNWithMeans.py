#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader

import pandas as pd

 #pre-processed file with normalized ratings by user
df = pd.read_csv(r"~/desenv/rcm-portal/dados/id_menu+id_usuario+mean_score.csv", delimiter=";")

max_score = df['score'].max()
reader = Reader(rating_scale=(0, max_score))

data = Dataset.load_from_df(df[['userId', 'itemId', 'score']], reader)
algo = KNNWithMeans(k=10, sim_options={'name': 'pearson_baseline', 'user_based': True})

trainset = data.build_full_trainset()

algo.fit(trainset)

print(algo.predict(7348, 244, 0.5))

x = df.tail(50)

for index, row in x.iterrows():
    print(row['userId'], row['itemId'], algo.predict(row['userId'], row['itemId'], row['score']))
    
#TODO: criar dataframe com as predicoes