#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()


# In[2]:


features = iris.data
labels = iris.target


# In[3]:


features[0]


# In[4]:


labels[0]


# In[5]:


clf = KNeighborsClassifier()
clf.fit(features,labels)


# In[11]:


preds = clf.predict([[31,1,1,1]])
preds


# In[ ]:




