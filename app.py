#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import module
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df = pd.read_csv('Housing_Dataset_Sample.csv')


# In[3]:


df.head()
df.describe()


# In[4]:


df.describe().T


# In[5]:


sns.distplot(df['Price'])


# In[6]:


sns.jointplot(x=df['Avg. Area Income'], y=df['Price'], data=df)


# In[7]:


sns.pairplot(df)


# In[8]:


X = df.iloc[:,:5]
y = df['Price']


# In[9]:


#split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54)


# In[10]:


#train model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
#check model
predictions = reg.predict(X_test)
predictions


# In[21]:


y_test.iloc[0]


# In[22]:


predictions[0]


# In[23]:


#check model performance
from sklearn.metrics import r2_score
r2_score(y_test, predictions)


# In[28]:


plt.scatter(y_test, predictions, alpha=0.1)


# In[26]:


sns.jointplot(x=y_test, y=predictions, data=df)


# In[ ]:




