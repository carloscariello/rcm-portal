#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:03:51 2018

@author: cariello
"""
import time
import datetime


from private.busca_dados import BuscaDados
from private.envia_dados import EnviaDados

from gera_score import GeraScore
from KNNWithMeans import KMeans

workspace =  work="~/desenv/rcm-portal/dados"


start = time.time()
    
a = BuscaDados(workspace)
a.connect()

b = GeraScore(workspace)
b.run()

c = KMeans(workspace)
c.run()

d = EnviaDados(workspace)
d.connect()

cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))

print("Total time: %s" % cv_time)