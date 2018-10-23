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
import numpy as np
#dimport mysql.connector
from sqlalchemy import create_engine

config = { 
        'host':'dirao.bb.com.br',
        'database':'pagina_inicial_2017',
        'user':'rcm97',
        'password':'secret'
        }

engine = create_engine('mysql+mysqlconnector://%s:%s@%s:3306/%s' % (config['user'], config['password'], config['host'], config['database']), echo=False)

sqlMenus = """
    SELECT m.id AS itemId, m.palavras_chave AS tags FROM menu m
    WHERE m.palavras_chave IS NOT NULL
	AND (m.dt_fim > NOW() OR m.dt_fim IS NULL )
    """

menuDf = pd.read_sql(sqlMenus, engine)

tagDict = {}
for index, row in menuDf.iterrows():
    for word in row['tags'].split(" "):
        if( len(word) > 1):
            tagDict[word] = 1

tagVector = []
for key in sorted(tagDict):
    if( len(key) > 2 ):
        tagVector.append(key.lower())

x = menuDf
for idx, val in enumerate(tagVector):
    print(idx, val)
    x["tag_%d" % idx] = menuDf['tags'].str.contains(val)

del x['tags']
tagDf = pd.DataFrame(dict(a=np.array(tagVector).tolist()))


tagDf = pd.Series(tagVector).to_frame('tag').reset_index(level=['index', 'tag'])
