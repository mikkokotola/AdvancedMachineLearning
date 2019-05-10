#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Advanced Course in Machine Learning
## Week 7
## Exercise 2 / Neural networks for sequences

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import LSTM

import keras


# In[2]:


# Read in the training data
# Path /home/mkotola/AdvancedML/Week7
dataFile = "seqtrain.csv"
data = pd.read_csv(dataFile, sep=",", header=None)
# N = 4000, d = 31


# In[3]:


# Read in the validation data
validDataFile = "seqvalid.csv"
validData = pd.read_csv(validDataFile, sep=",", header=None)


# In[4]:


y = data.iloc[:,29:33]
X = data.iloc[:,0:29]
y_valid = validData.iloc[:,29:33]
X_valid = validData.iloc[:,0:29]


# In[5]:


#print(X.mean(axis=0))
#print(X.std(axis=0))


# In[6]:


# Scale the test data
sc = StandardScaler()
X = sc.fit_transform(X)
X_valid = sc.transform (X_valid)


# In[7]:


#print(X.mean(axis=0))
#print(X.std(axis=0))


# In[8]:


#print(y)


# In[9]:


def codeYClassr(orig):  
    y = orig.copy()
    y['yclass'] = np.where(y.iloc[:,0]==1, 29, 0)
    y['yclass'] = np.where(y.iloc[:,1]==1, 30, y.iloc[:,4])
    y['yclass'] = np.where(y.iloc[:,2]==1, 31, y.iloc[:,4])
    y['yclass'] = np.where(y.iloc[:,3]==1, 32, y.iloc[:,4])
    return y


# In[10]:


y_cl = codeYClassr(y)
y_val_cl = codeYClassr(y_valid)
y_train = y_cl['yclass']
y_validate = y_val_cl['yclass']


# In[12]:


# Logistic regression
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y_train)


# In[13]:


pred = clf.predict(X_valid)


# In[15]:


result = y_validate.eq(pred)
print(np.sum(result)/result.size)


# In[16]:


# Logistic regression is able to classify the validation instances only 24 % correctly - which is worse than random
# guessing! Random guessing would result in 25 % correct labeling with 4 classes.


# In[18]:


# Keras model / MLP
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=29))
model.add(Dense(units=64, activation='relu', input_dim=64))
model.add(Dense(units=4, activation='softmax'))


# In[71]:


model.summary()


# In[19]:


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#model.compile(loss=keras.losses.categorical_crossentropy, 
#optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))


# In[20]:


model.fit(X, y, epochs=5, batch_size=32)


# In[21]:


loss_and_metrics = model.evaluate(X_valid, y_valid, batch_size=128)


# In[22]:


print(loss_and_metrics)


# In[23]:


# Keras model / CNN A
model_m = Sequential()
model_m.add(Conv1D(100, 3, activation='relu', input_shape=(29, 1)))
model_m.add(Conv1D(100, 3, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 3, activation='relu'))
model_m.add(Conv1D(160, 3, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(4, activation='softmax'))


# In[24]:


print(model_m.summary())


# In[25]:


# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model_m.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[26]:


X_exp = np.expand_dims(X, axis=2)


# In[27]:


X_valid_exp = np.expand_dims(X_valid, axis=2)


# In[28]:


#model.fit(X, y, epochs=5, batch_size=32)
#model.fit(X, y,
#              batch_size=32,
#              epochs=5,
#              validation_data=(X_valid, y_valid), shuffle=True)
model_m.fit(X_exp, y, batch_size=32, epochs=20)


# In[29]:


loss_and_metrics = model_m.evaluate(X_valid_exp, y_valid, batch_size=128)
print(loss_and_metrics)


# In[31]:


# Keras LSTM
model_lstm = Sequential()
model_lstm.add(Embedding(1024, output_dim=256))
model_lstm.add(LSTM(128))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(4, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model_m.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[66]:


# Keras LSTM
model_l = Sequential()

#model_l.add(LSTM(32, input_shape=(29,1)))  # returns a sequence of vectors of dimension 32

model_l.add(LSTM(32, return_sequences=True,
               input_shape=(29, 1)))  # returns a sequence of vectors of dimension 32
model_l.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model_l.add(LSTM(32))  # return a single vector of dimension 32
model_l.add(Dense(4, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model_l.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# In[67]:


model_l.summary()


# In[68]:


model_l.fit(X_exp, y, batch_size=16, epochs=10)


# In[70]:


loss_and_metrics = model_l.evaluate(X_valid_exp, y_valid, batch_size=16)
print(loss_and_metrics)

