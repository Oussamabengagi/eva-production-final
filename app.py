#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import json
import numpy as np
import pickle
import nltk
import tensorflow as tf
keras = tf.keras
from nltk.stem import WordNetLemmatizer
from keras.models import load_model


# In[2]:


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', 'r').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbotmodel.h5')
load_classifier = open("emotion_detector.pickle", "rb")
classifier = pickle.load(load_classifier)


# In[3]:


def clean_up_sentence(sentence):
    sentence_word = nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word


# In[4]:


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)


# In[5]:


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHHOLD = 0.25
    results =[[i, r] for i, r in enumerate(res) if r > ERROR_THRESHHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list


# In[6]:


def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result


# In[7]:


def preprocess(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    return words


# In[8]:


def get_features(word_list):
    return dict([(word, True) for word in word_list])


# In[9]:


def classify_emotion(classifier, text):
    features = get_features(preprocess(text))
    sentiment = classifier.classify(features)
    return sentiment


# In[10]:


def save_history(message, res):
    with open("conversation_history.txt", "a") as f:
        f.write("You: " + message + "\n")
        f.write("AI Therapist: " + res + "\n")


# In[11]:


import time

def slow_type(message):
    for char in message:
        print(char, end='', flush=True)
        time.sleep(0.1)


# In[12]:


import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[ ]:


app = Flask("__name__")
@app.route("/")
def loadPage():
	return "Welcome to EVA!"

@app.route("/eva", methods=['POST'])
def evamindmate():
    data = request.json
    message = data.get('message')
    ints = predict_class(message)
    res = get_response(ints, intents)
    return {"response":res}

if __name__ == '__main__':
    app.run()

# In[ ]:





# In[ ]:




