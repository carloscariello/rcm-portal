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
#dimport mysql.connector
from sqlalchemy import create_engine

config = { 
        'host':'dirao.bb.com.br',
        'database':'pagina_inicial_2017',
        'user':'rcm97',
        'password':'secret'
        }

engine = create_engine('mysql+mysqlconnector://%s:%s@%s:3306/%s' % (config['user'], config['password'], config['host'], config['database']), echo=False)

liste_hello = ['hello1','hello2']
liste_world = ['world1','world2']
df = pd.DataFrame(data = {'hello' : liste_hello, 'world': liste_world})
 
# Writing Dataframe to Mysql and replacing table if it already exists
df.to_sql(name='helloworld', con=engine, if_exists = 'replace', index=False)



data = pd.read_sql('SELECT * FROM helloworld', engine)