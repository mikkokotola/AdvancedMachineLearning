#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Advanced Course in Machine Learning
## Week 2
## Exercise 1

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# a)
def calcRHat (a, M):
    x = np.random.uniform(-2,2,M)
    noise = np.random.uniform(-1,1,M)
    y = 2*x+noise

    loss = (y - a*x)**2
    Rhat = sum(loss)/M
    return Rhat
    #return x, y, loss, Rhat


# In[3]:


a = 1
M = 100000

Rhats = list()
lowerA = -100
higherA = 100
axisX = range(lowerA, higherA+1)
for a in axisX:
    Rhats.append(calcRHat(a,M))
    #x, y, loss, RHat = calcRHat(a,M)


# In[4]:


sns.set_style("darkgrid")
plt.plot(axisX, Rhats)
#plt.xlim([-100, 100])
plt.xlabel('Alpha')
plt.ylabel('R_hat')
plt.title('Alpha vs. R_hat using Monte Carlo approximation')
plt.show()


# In[5]:


a = 2.5

Rhats = list()
lowerM = 1
higherM = 100000
axisX = range(lowerM, higherM+1)
for M in axisX:
    Rhats.append(calcRHat(a,M))


# In[6]:


plt.plot(axisX, Rhats)
#plt.xlim([-100, 100])
plt.xlabel('M')
plt.ylabel('R_hat')
plt.title('M vs. R_hat using Monte Carlo approximation')
plt.show()

