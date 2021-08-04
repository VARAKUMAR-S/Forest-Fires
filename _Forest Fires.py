#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Prepare a classification model using SVM for salary data


# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns


# In[12]:


forestfires = pd.read_csv("D:\\360DigiTMG\\Black Box Technique SVM\\HANDS ON MATERIAL\\Black Box Technique-SVM\\forestfires.csv")


# In[13]:


data = forestfires.describe()


# In[14]:


#Dropping the month and day columns
forestfires.drop(["month","day"],axis=1,inplace =True)


# In[15]:


#Normalising the data as there is scale difference
predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]


# In[16]:


def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)


# In[17]:


fires = norm_func(predictors)


# In[18]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)


# In[20]:


model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)


# In[21]:


np.mean(pred_test_linear==y_test) # Accuracy = 100%


# In[22]:


# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)


# In[23]:


np.mean(pred_test_poly==y_test) #Accuacy = 100%


# In[24]:


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)


# In[25]:


np.mean(pred_test_rbf==y_test) #Accuracy = 74.6%


# In[26]:


#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)


# In[27]:


np.mean(pred_test_sig==y_test) #Accuracy = 73%

