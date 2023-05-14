#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Embedding,Bidirectional
from keras.layers import LSTM
from keras.optimizers import SGD


# In[2]:


lemmatizer = WordNetLemmatizer()


# In[3]:


intents = json.loads(open('intents.json').read())


# In[4]:


words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']


# In[5]:


import nltk


# In[6]:


nltk.download('punkt')


# In[7]:


for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[8]:


nltk.download('wordnet')


# In[9]:


nltk.download('omw-1.4')


# In[10]:


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))


# In[11]:


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# In[12]:


training = []
output_empty = [0] * len(classes)


# In[13]:


for document in documents:
    bag=[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])


# In[14]:


print(len(training))


# In[15]:


random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


# In[16]:


print(len(training))


# In[17]:


model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))


# In[18]:


sgd = SGD(learning_rate=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['Accuracy'])
hist = model.fit(np.array(train_x),np.array(train_y),epochs=200, batch_size=5,verbose=1)
model.save('chatbotmodel.h5',hist)
print('done')


# In[19]:


model.summary()


# In[ ]:




