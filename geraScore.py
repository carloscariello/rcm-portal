#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 15:19:19 2018

@author: cariello
"""
"""
    Esse programa cria a pontuacao que mostra a relevancia de cada item pra cada usuario
    Para amenizar as diferencas entre os niveis de atividade de cada usuario, o score de
    cada acesso é dividido pelo seu score total.
"""


import pandas as pd
import math
from datetime import datetime

scoreFn = lambda x : math.exp( -( x**2 )/360 )

class GeraScore:
    
    work="~/desenv/rcm-portal/dados"
    input_csv = r"%s/user+item +team+ts_reg_20181206.csv" % work
    output_csv = r"%s/id_menu+id_usuario+mean_score.csv" % work
    
    def run(self):
        df = pd.read_csv(self.input_csv, delimiter=";")

        counts = df['userId'].value_counts()

        df = df[ ~ df['userId'].isin( counts[counts < 10].index ) ]
        
        del counts
        
        df['now'] = datetime.now()
        df['ts_reg'] = pd.to_datetime(df['ts_reg'], format='%Y-%m-%d %H:%M:%S')
        
        df['days'] = [int(i.days) for i in ( df['now'] - df['ts_reg'] )]
        
        df['score'] = df['days'].astype(float).map( scoreFn )
        
        del df['now']
        del df['days']
        del df['ts_reg']
        
        #"""
        #smoothing the user avg score
        sumByUser = df.groupby(['userId'])['score'].sum()
        sumByUser = sumByUser.to_frame().reset_index(level=['userId'])
        sumByUser.columns = ['userId', 'totalScore']
        join = pd.merge(df, sumByUser, on='userId')      
        join['score'] = join['score'] / join['totalScore']        
        x = join.groupby(['userId', 'itemId'])['score'].sum()
        x = x.to_frame().reset_index(level=['userId', 'itemId'])      
        df = x
        
        limiar = .05
        df['score'][ df['score'] < limiar ] = limiar

        #"""
        
        #df = df.groupby(['userId', 'itemId'])['score'].sum()
        #df = df.to_frame().reset_index(level=['userId', 'itemId'])

                
        df.to_csv(self.output_csv, sep=';', encoding='utf-8', index=False)
        
        return df

#g = GeraScore()
#df = g.run()



