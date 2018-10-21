#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader

import pandas as pd

output_csv= r'/home/cariello/desenv/rcm-portal/dados/%s' % 'predictions+cf+knnmeans.csv'

 #pre-processed file with normalized ratings by user
print("Loading CSV data")
df = pd.read_csv(r"~/desenv/rcm-portal/dados/id_menu+id_usuario+mean_score.csv", delimiter=";")

max_score = df['score'].max()
print("Max Score is %d" % max_score)
reader = Reader(rating_scale=(0, max_score))

data = Dataset.load_from_df(df[['userId', 'itemId', 'score']], reader)
algo = KNNWithMeans(k=10, sim_options={'name': 'pearson_baseline', 'user_based': True})

print("Building Trainset")
trainset = data.build_full_trainset()

print("Trainning")
algo.fit(trainset)

#x = algo.predict(7348, 244)[3]

predColumns = ['userId', 'itemId', 'score', 'est']

predictDf = pd.DataFrame([], columns=predColumns)

print("Generating predictions dataframe... (wait)")
for index, row in df.iterrows():
    userId = int (row['userId'])
    itemId = int(row['itemId'])
    score = row['score']
    prediction = algo.predict(row['userId'], row['itemId']) [3]
    predictDf = predictDf.append(  
                     pd.DataFrame( [ [ userId, itemId, score, prediction] ], columns=predColumns),
                     ignore_index=True
                     )
    
print("Building output CSV file (%s)" % output_csv)
predictDf.to_csv(output_csv, sep=';', encoding='utf-8', index=False)

print("Finished")