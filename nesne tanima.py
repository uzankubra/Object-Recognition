#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()


# In[3]:


X_train.shape


# In[4]:


X_test.shape


# In[5]:


y_train[:3]


# y_train ve y_test 2 boyutlu bir array olarak tutuluyor cifar10 verisetinde. 
# Biz bu verileri görsel olarak daha rahat anlamak için tek boyutlu hale getiriyoruz.
# 2 boyutlu bir arrayi (sadece tekbir boyutunda veri var diğer boyutu boş olan tabi) tekboyutlu hale geitrmek için reshape() kullanıyoruz..

# In[6]:


y_test = y_test.reshape(-1,)


# In[7]:


y_test 


# In[8]:


resim_siniflari = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[9]:


def plot_sample(X, y, index):  #
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])        
    plt.xlabel(resim_siniflari[y[index]])
    plt.show()


# In[10]:


plot_sample(X_test, y_test, 0)


# In[11]:


plot_sample(X_test, y_test, 1)


# In[12]:


plot_sample(X_test, y_test, 3)


# In[13]:


X_train = X_train / 255
X_test = X_test / 255


# ### Deep Learning Algoritmamızı CNN - Convolutional Neural Network Kullanarak Tasarlıyoruz:

# In[14]:


deep_learning_model = models.Sequential([
    # Bu kısımda fotoğraflardan tanımlama yapabilmek için özellikleri çıkarıyoruz
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    #Yukarıdaki özelliklerimiz ve training bilgilerine göre ANN modelimizi eğiteceğiz.
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[15]:


deep_learning_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# # Model eğitimi

# In[16]:


deep_learning_model.fit(X_train, y_train, epochs=5)


# In[17]:


deep_learning_model.evaluate(X_test,y_test)


# In[18]:


y_pred = deep_learning_model.predict(X_test)
y_pred[:3]


# In[19]:


y_predictions_siniflari = [np.argmax(element) for element in y_pred]
y_predictions_siniflari[:3]


# In[20]:


y_test[:3]


# In[21]:


plot_sample(X_test, y_test,0)


# In[22]:


resim_siniflari[y_predictions_siniflari[0]]


# In[23]:


plot_sample(X_test, y_test,1)


# In[ ]:


resim_siniflari[y_predictions_siniflari[1]]


# In[ ]:


plot_sample(X_test, y_test,2)


# In[ ]:


resim_siniflari[y_predictions_siniflari[2]]


# In[ ]:





# In[ ]:




