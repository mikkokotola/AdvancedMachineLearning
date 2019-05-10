#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Advanced Course in Machine Learning
## Week 6
## Exercise 3 / Automatic differentiation checkup

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import linalg as LA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import math
import sys

import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

sns.set_style("darkgrid")


# In[11]:


e = np.array([1,0])
print(err*e)


# In[12]:


def f(x):
    x1 = x[0]
    x2 = x[1]
    return (x1**2)*(math.sin((x1**2)+math.exp(x2)))

def findif_x1(x, err):
    e = np.array([1,0])
    return ((f(np.add(x, err*e)) - f(x))/err)

def findif_x2(x, err):
    e = np.array([0,1])
    return ((f(np.add(x, err*e)) - f(x))/err)   


# In[16]:


err = 0.00005
x = np.array([1,2])
findif1 = findif_x1(x, err)
findif2 = findif_x1(x, err)
print(findif1)
print(findif2)


# In[17]:


x1 = x[0]
x2 = x[1]
diff_x1 = 2*x1*(x1**2)*math.cos(((x1**2)+math.exp(x2)))
diff_x2 = x1*math.cos((x1**2)+math.exp(x2))
print(diff_x1)
print(diff_x2)

