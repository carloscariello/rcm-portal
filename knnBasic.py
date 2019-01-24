#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:01:15 2018

@author: cariello
"""

import surprise

import pandas as pd
from collections import defaultdict
import math




class KnnBasic:
    work="~/desenv/rcm-portal/dados"
    
    
    def __init__(self, workspace):
        if (workspace != None):
            self.work = workspace
        
        self.input_csv = r"%s/id_menu+id_usuario+mean_score.csv" % self.work
        self.output_csv = "%s/knn-basic-top-rated.csv" % self.work
    
    def get_top_n(self, predictions, n=10):
        '''From @NicolasHug
        '''
    
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
    
        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
    
        return top_n
    
    def run(self):
        df = pd.read_csv(self.input_csv, delimiter=";")
        
        lower_rating = df['score'].min()
        upper_rating = df['score'].max()
        
        print('Score range: {0} to {1}'.format(lower_rating, upper_rating))
        
        reader = surprise.Reader(rating_scale= (df['score'].min(), df['score'].max()))
        data = surprise.Dataset.load_from_df(df, reader)
        trainset = data.build_full_trainset()
        
        chosen_k =  math.ceil( math.sqrt( len(df['userId'].unique()) ) + 1)
        algo = surprise.KNNBasic(
                k=chosen_k, 
                sim_options={'name': 'pearson_baseline', 'user_based': True})
        algo.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        
        del df
        del reader
        
        top_n = self.get_top_n(predictions, n=10)
        
        df_top_rated = pd.DataFrame(columns=['userId', 'itemId', 'est'])
        
        
        for uid, user_ratings in top_n.items():
            for iid, est in user_ratings:
                df_top_rated.loc[len(df_top_rated)] = [uid, iid, est]
        
        
        df_top_rated.to_csv(self.output_csv, sep=';', encoding='utf-8', index=False)
        
        return df_top_rated

#validate = surprise.model_selection.cross_validate(alg, data, verbose = True)