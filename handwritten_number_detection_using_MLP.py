#!/usr/bin/env python
# coding: utf-8

# In[54]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[55]:


pip install tensorflow


# In[56]:


(x_train, y_train) ,(x_test, y_test)=keras.datasets.mnist.load_data()


# In[57]:


len(x_train)


# In[58]:


len(x_train)


# In[59]:


len(x_test)


# In[60]:


x_train[0].shape


# In[61]:


x_train[0]


# In[62]:


plt.matshow(x_train[0])


# In[63]:


y_train[0]


# In[64]:


y_train[:5]


# In[65]:


x_train.shape


# In[66]:


x_train=x_train/255
x_test=x_test/255


# In[67]:


x_train_flattened=x_train.reshape(len(x_train),28*28)
x_test_flattened=x_test.reshape(len(x_test),28*28)
x_test_flattened.shape
x_train_flattened.shape


# In[68]:


x_train_flattened[0]


# In[69]:


model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_flattened,y_train,epochs=10)


# In[70]:


model.evaluate(x_test_flattened,y_test)


# In[71]:


plt.matshow(x_test[0])


# In[72]:


y_pred=model.predict(x_test_flattened)


# In[73]:


y_pred[0]


# In[74]:


np.argmax(y_pred[0])


# In[75]:


y_pred_labels=[np.argmax(i) for i in y_pred]


# In[76]:


y_pred_labels[:5]


# In[77]:


y_test[:5]


# In[78]:


cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)


# In[79]:


cm


# In[80]:


pip install seaborn


# In[81]:


import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[82]:


model=keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
     keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train_flattened,y_train,epochs=5)


# In[83]:


model.evaluate(x_test_flattened,y_test)


# In[84]:


y_pred=model.predict(x_test_flattened)
y_pred_labels=[np.argmax(i) for i in y_pred]
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[85]:


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(200,activation='relu'),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=5)


# In[88]:


model.evaluate(x_test,y_test)


# In[87]:


y_pred=model.predict(x_test)
y_pred_labels=[np.argmax(i) for i in y_pred]
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)

import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('Truth')


# In[ ]:




