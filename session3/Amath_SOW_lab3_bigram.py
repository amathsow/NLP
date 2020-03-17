#!/usr/bin/env python
# coding: utf-8

# In[89]:


import io, sys, math, re
from collections import defaultdict
import numpy as np


# In[90]:


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


# In[91]:


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


# In[92]:


# LOAD DATA

train_data, vocab = load_data("train2.txt")
## FILL CODE 
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# rare words with <unk> in the dataset
train_data_ = remove_rare_words(train_data, vocab, mincount = 3)

print("load validation set")
valid_data, _ = load_data("valid2.txt")
## FILL CODE 
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# OOV with <unk> in the dataset
valid_data_ = remove_rare_words(valid_data, vocab, mincount = 3)


# In[99]:


# Function to build a bigram model

def build_bigram(data):
    unigram_counts = defaultdict(lambda:0)
    bigram_counts  = defaultdict(lambda: defaultdict(lambda: 0.0))
    total_number_words = 0

    ## FILL CODE
    # Store the unigram and bigram counts as well as the total 
    # number of words in the dataset
    for sentence in data:
        for word in sentence:
            unigram_counts[word] +=1
            total_number_words +=1
    for sentence in data:
        for i in range(1,len(sentence)):
            bigram_counts[sentence[i-1]][sentence[i]] +=1
            
    
    unigram_prob = defaultdict(lambda:0)
    bigram_prob = defaultdict(lambda: defaultdict(lambda: 0.0))

    ## FILL CODE
    # Build unigram and bigram probabilities from counts
    for sen in data:
        for w in sen:
            unigram_prob[w] = unigram_counts[w]/total_number_words
     
    for key1, sentence in enumerate(data):
        for key2, word in enumerate(sentence):
            if key2+1 < len(sentence):
                num = bigram_counts[data[key1][key2]][data[key1][key2+1]]
                denom = unigram_counts[data[key1][key2]]
                bigram_prob[data[key1][key2]][data[key1][key2+1]] = num/denom


    return {'bigram': bigram_prob, 'unigram': unigram_prob}


# In[100]:


# RUN TO BUILD BIGRAM MODEL

print("build bigram model")
model = build_bigram(train_data)


# In[101]:


def get_prob(model, w1, w2):
    assert model["unigram"][w2] != 0, "Out of Vocabulary word!"
    ## FILL CODE
    # Should return the probability of the bigram (w1w2) if it exists
    # Else it return the probility of unigram (w2) multiply by 0.4
    if model['bigram']!=0:
        return model['bigram'][w1][w2]
    else:
        return model['unigram'][w2] * 0.4
def perplexity(model, data):
    ## FILL CODE
    # follow the formula in the slides
    # call the function get_prob to get P(w2 | w1)
    perpl = 0
    T = 1
    
    for key1, sent in enumerate(data):
        T = 1.0
        perp_s = 1.0
        for key2, word in enumerate(sent):
            if key2+1 < len(data[key1]):
                if model['bigram'][data[key1][key2]][data[key1][key2+1]] == 0:
                    continue
                    
                perp_s *= model['bigram'][data[key1][key2]][data[key1][key2+1]]**(-1/T)
                T+=1
        perpl += perp_s
    return perpl


# In[102]:


# COMPUTE PERPLEXITY ON VALIDATION SET

print("The perplexity is", perplexity(model, valid_data))


# In[103]:


def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    i = 0
    while True :
        model['bigram'][sentence[i]].keys()
        x = list(model['bigram'][sentence[i]].keys())
        y = list(model['bigram'][sentence[i]].values())
        pred = np.random.choice(x, 1, p = y)

        sentence.append(pred[0])
        i+=1
        if pred[0] == '</s>':
            break
    return sentence


# In[104]:


# GENERATE A SENTENCE FROM THE MODEL

print("Generated sentence: ",generate(model))


# In[ ]:




