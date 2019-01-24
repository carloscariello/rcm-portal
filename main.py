#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:03:51 2018

@author: cariello
"""
import time
import datetime
import pandas as pd

#from private.busca_dados import BuscaDados
#from private.envia_dados import EnviaDados

from geraScore import GeraScore
from knnBasic import KnnBasic

workspace =  work="~/desenv/rcm-portal/dados"


start = time.time()
    
#a = BuscaDados(workspace)
#a.connect()

g = GeraScore()
df = g.run()

c = KnnBasic(workspace)
c.run()

#d = EnviaDados(workspace)
#d.connect()

cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))

print("Total time: %s" % cv_time)