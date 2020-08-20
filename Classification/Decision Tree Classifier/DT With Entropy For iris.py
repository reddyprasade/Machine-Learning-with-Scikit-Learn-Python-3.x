#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


path = "https://raw.githubusercontent.com/reddyprasade/Machine-Learning-Problems-DataSets/master/Classification/iris.csv"
df = pd.read_csv(path)
df.head()


# In[3]:


df.tail()


# In[4]:


X = df[['sepal_length','sepal_width','petal_length','petal_width']]
Y = df.species


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train,x_test,y_train,y_test = train_test_split(X,Y)


# In[7]:


from sklearn.tree import DecisionTreeClassifier


# In[32]:


DTC = DecisionTreeClassifier(criterion='entropy',max_features=2)


# In[33]:


DTC.fit(x_train,y_train)


# In[34]:


yhat = DTC.predict(x_test)


# In[35]:


importance  = DTC.feature_importances_


# In[36]:


for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[29]:


from sklearn import tree


# In[37]:


tree.plot_tree(DTC)


# In[38]:


plt.figure(figsize=(16,9))
tree.plot_tree(DTC, filled=True)
plt.show()


# In[ ]:





# In[ ]:




