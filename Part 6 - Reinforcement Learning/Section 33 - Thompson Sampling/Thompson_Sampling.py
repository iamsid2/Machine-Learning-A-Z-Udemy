#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 02:39:04 2018

@author: sid
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
ads_selected = []
no_of_times_reward_1 = [0] * d
no_of_times_reward_0 = [0] * d
total_reward = 0
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        theta_i = random.betavariate(no_of_times_reward_1[i]+1,no_of_times_reward_0[i]+1)
        if theta_i > max_random:
            max_random = theta_i
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        no_of_times_reward_1[ad] += 1
    else:
        no_of_times_reward_0[ad] += 1
    total_reward = total_reward + reward

#Visualising UCB
plt.hist(ads_selected)
plt.title('Histogram of ad selection')
plt.xlabel('Ads')
plt.ylabel('No. of times selected')
plt.show()
        
        

