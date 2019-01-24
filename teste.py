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


work="~/desenv/rcm-portal/dados"
input_csv = r"%s/user+item +team+ts_reg_20181206.csv" % work
output_csv = r"%s/id_menu+id_usuario+mean_score.csv" % work



df = pd.read_csv(input_csv, delimiter=";")

counts = df['userId'].value_counts()

res = df[ ~ df['userId'].isin( counts[counts < 10].index ) ]

del counts


