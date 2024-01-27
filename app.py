# import module
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


df = pd.read_csv('Housing_Dataset_Sample.csv')


# In[8]:


df.head()
df.describe()


# In[9]:


df.describe().T


# In[13]:


sns.distplot(df['Price'])


# In[15]:


sns.jointplot(x=df['Avg. Area Income'], y=df['Price'], data=df)


# In[16]:


sns.pairplot(df)


# In[21]:


X = df.iloc[:,:5]
y = df['Price']


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=54)


# In[ ]:




