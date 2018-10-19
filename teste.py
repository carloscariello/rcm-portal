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

from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader

import pandas as pd

#pre-processed file with normalized ratings by user
df = pd.read_csv(r"~/desenv/rcm-portal/dados/id_menu+id_usuario+mean_score.csv", delimiter=";")
