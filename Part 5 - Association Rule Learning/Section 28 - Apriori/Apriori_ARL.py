#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 03:00:04 2018

@author: sid
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transaction = []
for i in range(7501):
    transaction.append([str(dataset.values[i, j]) for j in range (20)])
    
#Training Apriori in the dataset
from apyori import apriori
rules = apriori(transaction,min_support=0.003, min_confidence=0.2, min_lift=3,min_length=2)

#Visualising 
results = list(rules)