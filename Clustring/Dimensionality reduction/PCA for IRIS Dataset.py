#!/usr/bin/env python
# coding: utf-8

# ### Principal component analysis (PCA)
# 
# * PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance. In scikit-learn, PCA is implemented as a transformer object that learns `n`  components in its fit method, and can be used on new data to project it on these components.
# 
# PCA centers but does not scale the input data for each feature before applying the SVD. The optional parameter whiten=True makes it possible to project the data onto the singular space while scaling each component to unit variance. This is often useful if the models down-stream make strong assumptions on the isotropy of the signal: this is for example the case for Support Vector Machines with the RBF kernel and the K-Means clustering algorithm.
# 
#     Below is an example of the iris dataset, which is comprised of 4 features, projected on the 2 dimensions that explain most variance:

# In[2]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas ')
get_ipython().system('pip install matplotlib ')
get_ipython().system('pip install scikit-learn')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[5]:


iris = datasets.load_iris()


# In[6]:


iris.keys()


# In[7]:


X = iris.data
y = iris.target


# In[8]:


target_names = iris.target_names


# In[9]:


X.shape # 150 rows and 4 columns 


# In[10]:


pca = PCA(n_components=2) # 150 rows and 2 columns


# In[11]:


X_r = pca.fit_transform(X)
X_r.shape


# In[23]:


lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X,y).transform(X)


# In[14]:


# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))


# In[24]:


plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()


# In[ ]:




