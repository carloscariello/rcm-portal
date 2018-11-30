#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader

import pandas as pd

class KMeans:
    work = '/home/cariello/desenv/rcm-portal/dados'
    output_csv= ''
    
    def __init__(self, workspace):
        if (workspace != None):
            self.work = workspace
        
        self.output_csv = r"%s/predictions+cf+knnmeans.csv" % self.work
    
    def run(self):
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
            
        print("Building output CSV file (%s)" % self.output_csv)
        predictDf.to_csv(self.output_csv, sep=';', encoding='utf-8', index=False)
        
        print("Finished")
        
c = KMeans(None)
c.run()