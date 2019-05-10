#!/usr/bin/env python
# coding: utf-8

# In[46]:


## Advanced Course in Machine Learning
## Week 4
## Exercise 3 / Stochastic neighbor embedding

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

sns.set_style("darkgrid")


# In[2]:


x_train, t_train, x_test, t_test = mnist.load()


# In[3]:


print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


# In[4]:


x_train_s = x_train[0:1000, :]
t_train_s = t_train[0:1000]
print(x_train_s.shape)
print(t_train_s.shape)


# In[5]:


#np.savetxt("x_train_s_1000.csv", x_train_s, delimiter=",")


# In[6]:


#print(x_train_s[1,:])


# In[66]:


# Initialize z
mu, sigma = 0, 0.1 # mean and standard deviation
arraySize = (1000, 2)
z = np.random.normal(mu, sigma, arraySize)


# In[67]:


print(z.shape)
print(z)


# In[22]:


def calc_pairwise2(matrix):
    l1dist = pairwise_distances(matrix, metric='l1')
    l1distSquared = l1dist**2
    return l1distSquared, l1dist

def calc_pairwise(matrix, sigma):
    var = 2*(sigma**2)
    height = len(matrix[:,0])
    width = len(matrix[0,:])
    dist = np.zeros(shape=(height,height))
    for i in range(height):
        for j in range(height):
            if (i<j):
                d2 = np.log(np.sum((np.subtract(matrix[i,:], matrix[j,:])**2)/var))
                dist[i, j] = d2
                dist[j, i] = d2
    return dist.astype(np.float64)

def calc_p(d2):
    p = np.zeros(shape=(d2.shape))
    for k in range(len(d2[:,0])):
        d2[k,k] = sys.float_info.max
    #print('D2 after placement of max')
    #print(d2)
    for i in range(len(d2[:,0])):
        for j in range(len(d2[:,0])):
            upper = np.exp(-d2[i,j])
            lower = np.sum(np.exp(-d2[i]))-upper
            p[i,j] = upper/lower
    return p.astype(np.float64)


# In[23]:


# Standardize x_train
x_train_st = StandardScaler().fit_transform(x_train_s)
print(x_train_st)


# In[24]:


pca = decomposition.PCA(n_components=50)
pca.fit(x_train_st)
x_pc = pca.transform(x_train_st)
print(x_pc.shape)
#print(x_pc)


# In[69]:


print(x_pc.shape)
print(z.shape)


# In[70]:


sigma_orig = 1000
sigma_embed = 1
# Calculate the pairwise distances between samples in original space
dist_orig = calc_pairwise(x_pc, sigma_orig)
# Calculate the pairwise distances between samples in embedding space
dist_embed = calc_pairwise(z, sigma_embed)


# In[71]:


print(dist_orig.shape)
print(dist_orig)


# In[72]:


print(dist_embed.shape)
print(dist_embed)


# In[73]:


p_orig = calc_p(dist_orig)
p_embed = calc_p(dist_embed)


# In[74]:


print(p_orig)


# In[75]:


print(p_embed)


# In[76]:


def loss(p, q):
    return np.nansum(np.multiply(p, np.log(np.divide(p, q))))


# In[77]:


p_embed[p_embed == 0] = np.nan


# In[78]:


print(p_embed)


# In[79]:


p = p_orig
q = p_embed


# In[80]:


loss_1 = loss(p, q)


# In[81]:


print(loss_1)


# print(z.shape[0])
# s = np.zeros(shape=(z.shape[0],1))
# print(s)
#              

# In[113]:


def gradient (p, q, z):
    print('Shape of p ', str(p.shape))
    print('Shape of q ', str(q.shape))
    print('Shape of z ', str(z.shape))
    print('Length of z col 0')
    print(len(z[:,0]))
    print('Length of z col 1')
    print(len(z[:,0]))
    grad = np.zeros(shape=z.shape)
    for i in range(len(p[:,0])):
        print(i)
        s = np.zeros(shape=(z.shape))
        for j in range(len(p[0,:])):
            print(j)
            pq = p[i,j] - q[i,j] + p[j,i] - q[j,i]
            pq = np.asmatrix(pq)
            pq = pq.transpose()
            subt = np.subtract(z[:,i], z[:,j])
            s = np.add(s, (np.multiply(subt, pq)))
        grad[i,:] = s


# In[114]:


epochs = 5
initStepSize = 0.5
tau = 0.0005
lossHistory = []

lossHistory.append(loss_1)

stepSize = initStepSize
t = 0  

# Main loop
for epoch in np.arange(0, epochs):
    # initialize the total loss for the epoch
    epochLoss = []    
    
    t = t + 1
    
    gradientB = gradient(p, q, z)

    z -= stepSize * gradientB 
    stepSize = initStepSize / (1 + initStepSize * tau * t)
    dist_embed = calc_pairwise(z, sigma_embed)
    q = calc_p(dist_embed)
            
    lossHistory.append(loss(p, q))


# In[115]:


print(x_pc.shape)
print(x_pc)


# In[116]:


# Visualize PCA with 2 dimensions
plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
sns.scatterplot(x_pc[:, 0], x_pc[:, 1], hue=t_train_s, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right', fontsize='x-small')
plt.title('PCA, 2 components')
plt.show()


# In[47]:


# Compare with sklearn.manifold.tsne
X_embedded = TSNE(n_components=2).fit_transform(x_train_s)
print(X_embedded.shape)


# In[55]:


plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=t_train_s, alpha=0.5)
#origin = [0], [0] # origin point
#Plot the principal axis
#plt.quiver(*origin, w[0,0], w[1,0], color=['g'], scale=1, label='W')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right', fontsize='x-small')
plt.title('TSNE embedded space')
plt.show()


# In[ ]:




