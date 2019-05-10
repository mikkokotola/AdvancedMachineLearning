#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Advanced Course in Machine Learning
## Week 1
## Exercise 3

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## a)
dataFile = "ex_1_data.csv"
data = pd.read_csv(dataFile, header=None)
# N = 200, D = 5
dataOrig = data.copy()

# Standardize the data
data = pd.DataFrame(StandardScaler().fit_transform(data))
dataCopy = data.copy() # Do a comparison of the phases with the SKlearn PCA to double check


# In[2]:


## b)
covarianceMatrix = data.cov()
print('\nCovariance matrix: \n')
print(covarianceMatrix)


# In[3]:


eigenval, eigenvect = np.linalg.eig(covarianceMatrix)
eigenvalPD = pd.DataFrame(eigenval, columns=['eigval'])
eigenvectPD = pd.DataFrame(eigenvect)

eigenvalPD['origIndex'] = eigenvalPD.index
print(eigenvalPD)


# In[4]:


print(eigenvectPD)


# In[5]:


eigenvalPD = eigenvalPD.sort_values(by=['eigval'], ascending=False)
print(eigenvalPD)


# In[6]:


eigenvectPD = eigenvectPD.transpose()
print(eigenvectPD)


# In[7]:


eigenvectPD = eigenvectPD.reindex(eigenvalPD.index)
print(eigenvectPD)


# In[8]:


eigenvectPD = eigenvectPD.reset_index(drop=True)
#eigenvectPD = eigenvectPD.transpose()
print(eigenvectPD)


# In[9]:


projections = list()
for i in range(len(eigenvectPD.iloc[0])):
    projections.append(pd.DataFrame(np.dot(eigenvectPD.iloc[0:(i+1),:],data.transpose())))
#print (projections)

for i in range(len(projections)):
     projections[i] = projections[i].transpose()
    
print(projections)


# In[10]:


sns.scatterplot(projections[4].iloc[:,0], projections[4].iloc[:,1]).plot()
plt.title('Plot of 1. vs. 2. eigenvector projection of source data')
plt.xlabel('First eigenvector projection')
plt.ylabel('Second eigenvector projection')


# In[11]:


print (len(projections))
for i in range(len(projections)):
    #print(projections[i])
    projections[i] = pd.DataFrame(projections[i], columns=None)
    #projectionDF.append(pd.Series(projections[i]))
#print(dataOrig)
print(projections)


# In[12]:


#print(eigenvect)
#transpEigenvect = eigenvect.transpose()
#print(projections[0])
#reconstruct = np.dot(transpEigenvect[:,0], projections[0])
#print('1 dim\n')
#print(np.asmatrix(eigenvect[0:0]).transpose())
#print('2 dim\n')
#print(np.asmatrix(eigenvect[0:1]).transpose())

reconstructs = list()
for i in range(len(eigenvect)):
    reconstructs.append(np.dot(eigenvectPD.iloc[0:(i+1),:].transpose(), projections[i].transpose()).transpose())
#reconstruct = np.dot(eigenvect[:,0], projections[0].transpose())
print (len(reconstructs))
print(reconstructs[1])
       


# In[13]:


losses = list()

for rec in reconstructs:
    lossMatrix = dataOrig.sub(rec)
    print(lossMatrix)
    lossMatrix = lossMatrix**2
    print(lossMatrix)
    losses.append(lossMatrix.values.sum())
print(losses)


# In[14]:


sns.scatterplot(range(1,6), losses).plot()
plt.title('Plot of reconstruct squared error loss with principal components 1-5')
plt.xlabel('Number of principal components')
plt.ylabel('Reconstruct squared error loss')


# In[ ]:





# In[15]:


pca = PCA(n_components=5)
pca.fit(dataCopy)
components = pca.fit_transform(dataCopy)
cov = pca.get_covariance()
params = pca.get_params()
eigenvalues = pca.explained_variance_ 
print('Eigenvalues in descending order:')
print(eigenvalues)
eigenvalSquared = eigenvalues**2
print(eigenvalSquared)


# In[16]:


sns.scatterplot(components[:,0], components[:,1]).plot()
plt.title('PCA 1. vs. 2. component projection of source data')
plt.xlabel('First eigenvector projection')
plt.ylabel('Second eigenvector projection')


# In[17]:


max_comp=5
start=1
error_record=[]
for i in range(start,max_comp+1):
    pca = PCA(n_components=i)
    pca2_results = pca.fit_transform(dataCopy)
    pca2_proj_back=pca.inverse_transform(pca2_results)
    lossMatrix = dataOrig.sub(pca2_proj_back, axis='columns')**2
    error_record.append(lossMatrix.values.sum())

fig, ax1 = plt.subplots()
ax1.plot(error_record,'r')
ax1.set_xlabel('Principal components used')
ax1.set_ylabel('Squared error of reconstruct', color='r')
ax1.tick_params('y', colors='r')

ax2 = ax1.twinx()
ax2.plot(eigenvalues, 'b')
ax2.set_ylabel('Eigenvalue', color='b')
ax2.tick_params('y', colors='b')

fig.tight_layout()
plt.title("Reconstruct error of PCA compared to Eigenvalue")
plt.xticks(range(len(error_record)), range(start,max_comp+1))
plt.xlim([-1, len(error_record)])
plt.show()

