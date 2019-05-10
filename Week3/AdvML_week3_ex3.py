#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Advanced Course in Machine Learning
## Week 3
## Exercise 3 / Spectral clustering

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.cluster import KMeans


# In[2]:


# Read in the  data
dataFile = "exercise3data.csv"
data = pd.read_csv(dataFile, sep=",", header=None)
# N = 120, D = 2


# In[3]:


# print(data)


# In[4]:


distances = scipy.spatial.distance.cdist(data, data, metric='euclidean')
distMatrix = pd.DataFrame(distances)
print(distMatrix)


# In[5]:


sns.set_style("darkgrid")
plt.plot(data.iloc[:,0], data.iloc[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of source data')
plt.show()


# In[6]:


sns.set_style("darkgrid")
sns.scatterplot(data.iloc[:,0], data.iloc[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of source data')
plt.show()


# In[7]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(data)


# In[8]:


kmeans.labels_
#kmeans.predict([[0, 0], [12, 3]])


# In[9]:


kmeans.cluster_centers_


# In[10]:


sns.scatterplot(data.iloc[:,0], data.iloc[:,1], hue=kmeans.labels_)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of source data with SKLearn K-means labels')
plt.show()


# In[11]:


# Adjacency matrices
e = 0.5
adjMatrixE = distMatrix.apply(lambda x : x <= e)
np.fill_diagonal(adjMatrixE.values, 0)
print(adjMatrixE)


# In[12]:


A = 8
adjMatrixA = pd.DataFrame(0, index=np.arange(0, 120), columns=np.arange(0, 120), dtype=bool)
closestA = pd.DataFrame()
for i in range(len(distMatrix.iloc[:,0])):
    closestA.insert(i, i, distMatrix.nsmallest(A+1, i).iloc[:,i].drop(distMatrix.index[i]).index)
    for j in range(len(distMatrix.iloc[0,:])):
        #print(j)
        #print(closestA[i])
        if (j in closestA[i].values):
            adjMatrixA.at[i,j] = 1
            adjMatrixA.at[j,i] = 1        


# In[13]:


print(adjMatrixA)


# In[14]:


# Create diagonal matrix D
D_E = pd.DataFrame(0, index=np.arange(0, 120), columns=np.arange(0, 120))
D_A = pd.DataFrame(0, index=np.arange(0, 120), columns=np.arange(0, 120))

for i in range(120):
    D_E.at[i,i] = adjMatrixE.iloc[i,:].sum() 
    D_A.at[i,i] = adjMatrixA.iloc[i,:].sum()


# In[15]:


# Laplacians
L_E = D_E - adjMatrixE
L_A = D_A - adjMatrixA


# In[16]:


print(L_E)


# In[17]:


print(L_A)


# In[18]:


w_E, v_E = LA.eig(L_E)
w_A, v_A = LA.eig(L_A)


# In[19]:


eigenvalPD_E = pd.DataFrame(w_E, columns=['eigval'])
eigenvectPD_E = pd.DataFrame(v_E)

eigenvalPD_E = eigenvalPD_E.sort_values(by=['eigval'], ascending=True)
eigenvectPD_E = eigenvectPD_E.transpose()
eigenvectPD_E = eigenvectPD_E.reindex(eigenvalPD_E.index)
eigenvectPD_E = eigenvectPD_E.reset_index(drop=True)
# Eigenvectors in order on rows of the dataframe, smallest at the top


# In[20]:


print(eigenvalPD_E)


# In[21]:


print(eigenvectPD_E)


# In[22]:


x = range(120)
plt.plot(x, eigenvectPD_E.iloc[0,:])
plt.plot(x, eigenvectPD_E.iloc[1,:])
plt.plot(x, eigenvectPD_E.iloc[2,:])
plt.plot(x, eigenvectPD_E.iloc[3,:])
plt.legend(['Eigenvect 1', 'Eigenvect 2', 'Eigenvect 3', 'Eigenvect 4'], loc='upper right')
plt.xlabel('x')
plt.ylabel('Eigenvectors')
plt.title('Eigenvector of Laplacian using e < {}'.format(e))
plt.show()


# In[23]:


eigenvalPD_A = pd.DataFrame(w_A, columns=['eigval'])
eigenvectPD_A = pd.DataFrame(v_A)

eigenvalPD_A = eigenvalPD_E.sort_values(by=['eigval'], ascending=True)
eigenvectPD_A = eigenvectPD_A.transpose()
eigenvectPD_A = eigenvectPD_A.reindex(eigenvalPD_E.index)
eigenvectPD_A = eigenvectPD_A.reset_index(drop=True)
# Eigenvectors in order on rows of the dataframe


# In[24]:


x = range(120)
plt.plot(x, eigenvectPD_A.iloc[0,:])
plt.plot(x, eigenvectPD_A.iloc[1,:])
plt.plot(x, eigenvectPD_A.iloc[2,:])
plt.plot(x, eigenvectPD_A.iloc[3,:])
plt.legend(['Eigenvect 1', 'Eigenvect 2', 'Eigenvect 3', 'Eigenvect 4'], loc='lower left')
plt.xlabel('x')
plt.ylabel('Eigenvectors')
plt.title('Eigenvector of Laplacian using A = {} closest neighbors'.format(A))
plt.show()


# In[25]:


M = 4


# In[26]:


tr_E = eigenvectPD_E.iloc[0:M,:].transpose()
tr_A = eigenvectPD_A.iloc[0:M,:].transpose()


# In[27]:


for i in range(M):
    for j in range(M):
        if (i != j and i < j):
            sns.scatterplot(tr_E.iloc[:,i], tr_E.iloc[:,j])
            l1 = 'Eigenvect {}'.format(i)
            l2 = 'Eigenvect {}'.format(j)
            plt.xlabel(l1)
            plt.ylabel(l2)
            plt.title('Scatter plot of transformed data with e < 0.5')
            #plt.legend([l1 , l2])
            plt.show()


# In[28]:


for i in range(M):
    for j in range(M):
        if (i != j and i < j):
            sns.scatterplot(tr_A.iloc[:,i], tr_A.iloc[:,j])
            l1 = 'Eigenvect {}'.format(i)
            l2 = 'Eigenvect {}'.format(j)
            plt.xlabel(l1)
            plt.ylabel(l2)
            plt.title('Scatter plot of transformed data with A = {} nearest neighb.'.format(A))
            #plt.legend([l1 , l2])
            plt.show()


# In[29]:


kmeans_E = KMeans(n_clusters=2, random_state=0).fit(tr_E)
kmeans_A = KMeans(n_clusters=2, random_state=0).fit(tr_A)


# In[30]:


print(data.shape)
print(tr_E.shape)
print(tr_A.shape)


# In[31]:


print(kmeans_A.labels_)


# In[32]:


sns.scatterplot(data.iloc[:,0], data.iloc[:,1], hue=kmeans_E.labels_)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of source data with distance e < {} labels'.format(e))
plt.show()


# In[33]:


sns.scatterplot(data.iloc[:,0], data.iloc[:,1], hue=kmeans_A.labels_)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of source data with A = {} nearest neighb. labels'.format(A))
plt.show()

