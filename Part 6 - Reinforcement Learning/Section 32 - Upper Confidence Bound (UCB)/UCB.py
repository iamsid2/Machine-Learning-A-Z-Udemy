#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:17:28 2018

@author: sid
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
ads_selected = []
no_of_ad_selection = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (no_of_ad_selection[i] > 0):
            average_reward = sum_of_rewards[i] / no_of_ad_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / no_of_ad_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    no_of_ad_selection[ad] = no_of_ad_selection[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

#Visualising UCB
plt.hist(ads_selected)
plt.title('Histogram of ad selection')
plt.xlabel('Ads')
plt.ylabel('No. of times selected')
plt.show()
        
        

