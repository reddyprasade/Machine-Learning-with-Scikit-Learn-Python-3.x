#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install pandas-profiling')
get_ipython().getoutput('pip install scikit-learn')


# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


path = "https://raw.githubusercontent.com/reddyprasade/Machine-Learning-Problems-DataSets/master/Classification/Breast%20cancer%20wisconsin.csv"


# In[33]:


data = pd.read_csv(path);

print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(data.shape))


# In[34]:


data.head()


# In[35]:


data.tail()


# In[36]:


data.drop(['Unnamed: 0'],axis =1, inplace =True)


# In[37]:


data.info()


# In[38]:


diagnosis_all = list(data.shape)[0]
diagnosis_categories = list(data['diagnosis'].value_counts())

print("\n \t The data has {} Rows I ahAve among  diagnosis, {} malignant and {} benign.".format(diagnosis_all, 
                                                                                 diagnosis_categories[0], 
                                                                                 diagnosis_categories[1]))


# In[39]:


sns.countplot(diagnosis_categories)


# In[40]:


features_mean= list(data.columns[1:11])
features_mean


# In[41]:


plt.figure(figsize=(30,15))
sns.heatmap(data[features_mean].corr(),annot=True,square=True,cmap="coolwarm")


# In[42]:


from pandas.plotting import scatter_matrix


# In[43]:


color_dir = {"M":'green','B':'blue'}
colors = data['diagnosis'].map(lambda x:color_dir.get(x))

scatter_matrix(data[features_mean],c=colors,alpha=0.9,figsize=(20,20))
plt.show()


# In[50]:


M= data[data['diagnosis']=='M'][feature].count()
B= data[data['diagnosis']=='M'][feature].count()


# In[53]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import time


# In[54]:


from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()


# In[55]:


data['diagnosis'] = Le.fit_transform(data['diagnosis'])


# In[56]:


X = data.loc[:,features_mean]
y = data['diagnosis']
y


# In[57]:


X.isna().sum()


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[59]:


from sklearn.linear_model import LogisticRegression


# In[64]:


LR = LogisticRegression(max_iter=200)


# In[65]:


LR.fit(X_train,y_train)


# In[72]:


Train_Score = LR.score(X_train,y_train)
Train_Score


# In[74]:


Test_score = LR.score(X_test,y_test)
Test_score


# In[77]:


yhat = LR.predict(X_test)


# In[78]:


pd.DataFrame({"Actual Data":y_test,
             "New_predication":yhat})


# In[79]:


log_loss = LR.predict_proba(X_train)


# In[80]:


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# In[81]:


cm = confusion_matrix(y_test,yhat)


# In[82]:


cm


# In[83]:


print(classification_report(y_test,yhat))


# In[85]:


accuracy_score(y_test,yhat)


# ### KNN 

# In[86]:


from sklearn.neighbors import KNeighborsClassifier


# In[130]:


Knn = KNeighborsClassifier(n_neighbors=10)


# In[131]:


Knn.fit(X_train,y_train)


# In[132]:


Train_score = Knn.score(X_train,y_train)


# In[133]:


Train_Score


# In[134]:


Test_score = Knn.score(X_test,y_test)


# In[135]:


Test_score


# In[136]:


Knn.classes_


# In[137]:


Knn.predict_proba(X_train)


# In[ ]:




