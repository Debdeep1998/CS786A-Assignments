# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:10:38 2022

@author: Himanshu Lal
"""
#%%

import torch
import seaborn as sns

#%%
def plot_histogram_accuracy(pred_img, true_img, thresold = 10):
    
    hm = torch.abs(torch.sum(pred_img - true_img, dim = 0))
    sns.heatmap(hm, center = thresold)

#%%

pred_img = torch.randn(3, 5, 5)
true_img = torch.randn(3, 5, 5)

#%%