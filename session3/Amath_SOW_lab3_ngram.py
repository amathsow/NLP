#!/usr/bin/env python
# coding: utf-8

# In[64]:


import io, sys, math, re
from collections import defaultdict
import numpy as np


# In[65]:


# dataloader

def load_data(filename):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    vocab = defaultdict(lambda:0)
    for line in fin:
        sentence = line.split()
        data.append(sentence)
        for word in sentence:
            vocab[word] += 1
    return data, vocab


# In[66]:


def remove_rare_words(data, vocab, mincount):
    ## FILL CODE
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    for key1,sentence in enumerate(data):
        for key2,word in enumerate(sentence):
            if word not in vocab.keys():
                data[key1][key2]='<unk>'
            elif vocab[word] < mincount:
                data[key1][key2]='<unk>'

    return data


# In[67]:


# LOAD DATA

print("load training set")
train_data, vocab = load_data("train.txt")

## FILL CODE
# Same as bigram.py
train_data_ = remove_rare_words(train_data, vocab, mincount = 3)

print("load validation set")
valid_data, _ = load_data("valid.txt")
## FILL CODE
# Same as bigram.py
valid_data_ = remove_rare_words(valid_data, vocab, mincount = 3)


# In[68]:


def build_ngram(data, n):
    total_number_words = 0
    counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    for sentence in data:
        sentence = tuple(sentence)
        ## FILL CODE
        # dict can be indexed by tuples
        # store in the same dict all the ngrams
        # by using the context as a key and the word as a value
        total_number_words += len(sentence)
        for ngram in range(1,n+1):
            for i in range(len(sentence)-ngram+1):
                counts[sentence[i:(ngram+i-1)]][sentence[ngram+i-1]]+=1

    prob  = defaultdict(lambda: defaultdict(lambda: 0.0))
    ## FILL CODE
    # Build the probabilities from the counts
    # Be careful with how you normalize!
    for cont in counts.keys():
        for target in counts[cont].keys():
            prob[cont][target] = counts[cont][target]/(sum(counts[cont].values()))

    return prob


# In[69]:


# RUN TO BUILD NGRAM MODEL

n = 2
print("build ngram model with n = ", n)
model = build_ngram(train_data, n)


# In[70]:


def get_prob(model, context, w):
    ## FILL CODE
    # code a recursive function over 
    # smaller and smaller context
    # to compute the backoff model
    # Bonus: You can also code an interpolation model this way
    if model[context][w]!=0:
        return model[context][w]
    else:
        return get_prob(model,context[1:],w)*0.4

def perplexity(model, data, n):
    ## FILL CODE
    # Same as bigram.py
    perp = []
    for sentence in data:
        sentence = tuple(sentence)
        perps = 1.0
        for i in range(len(sentence)):
            if (n+i-1) < len(sentence):
                perps *= 1.0/get_prob(model, sentence[i:(n+i-1)], sentence[n+i-1]) 
        perp.append(perps**(1/len(sentence)))
    return np.mean(perp)


# In[71]:


# COMPUTE PERPLEXITY ON VALIDATION SET

print("The perplexity is", perplexity(model, valid_data, n))


# In[72]:


def get_proba_distrib(model, context):
    ## FILL CODE
    # code a recursive function over context
    # to find the longest available ngram 
    if sum(model[context].values()) !=0:
        return context
    else:
        return get_proba_distrib(model,context[1:])

def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    while True :
        x = list(model[(sentence[-1], )].keys())
        y = list(model[(sentence[-1], )].values())
        pred = np.random.choice(x, 1, p = y)
        sentence.append(pred[0])
        
        if pred[0] == '</s>':
            break
        
    return sentence


# In[73]:


# GENERATE A SENTENCE FROM THE MODEL

print("Generated sentence: ",generate(model))


# In[ ]:




