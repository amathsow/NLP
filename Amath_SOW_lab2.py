#!/usr/bin/env python
# coding: utf-8

# In[196]:


import io, sys
import numpy as np
from heapq import *


# In[197]:


def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
        
    return data


# In[198]:


# Loading word vectors

print('')
print(' ** Word vectors ** ')
print('')

word_vectors = load_vectors('wiki.en.vec')


# In[199]:


## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    num = u.dot(v)
    denom = np.linalg.norm(u)*np.linalg.norm(v)
    return num/denom 

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search


# In[200]:


# compute similarity between words

print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))


# In[201]:


## Functions for nearest neighbors
def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = ''
    for word in word_vectors:
        if word not in exclude_words:
            sim = cosine(x,word_vectors[word])
        if sim > best_score:
            best_score = sim
            best_word = word
    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []
    best_score = -1
    for word, vector in vectors.items():
        if (x!=vector).all():
            sim = cosine(x,vector)
            heappush(heap,(sim,word))
        if len(heap)> k:
            heappop(heap)

    return [heappop(heap) for i in range(len(heap))][::-1]


# In[202]:


# looking at nearest neighbors of a word

print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print(word + '\t%.3f' % score)


# In[203]:


def analogy(a,b,c,word_vectors):
    ## FILL CODE
    a, b, c= a.lower(), b.lower(), c.lower () 
    # find the word embeddings for word_a, word_b, word_c 
    e_a, e_b, e_c = word_vectors[a], word_vectors[b], word_vectors[c] 
    words = word_vectors.keys() 
    max_cosine_sim = -np.inf 
    best_word = None 
    # search for d in the whole word vector set 
    for w in words:
        # ignore input words 
        if w in [a, b, c]: 
            continue 
        # Compute cosine similarity between the vectors u and v 
        cos_sim = cosine(e_b - e_a, word_vectors[w] - e_c) 
        if cos_sim> max_cosine_sim: 
            max_cosine_sim = cos_sim 
            # update word_d 
            best_word = w
    return best_word


# In[204]:


# Word analogies

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))


# In[205]:


print('king - man + woman = ' + analogy('king', 'man', 'woman', word_vectors))


# In[206]:


## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))


# In[207]:


## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    elt_a = 0
    elt_b = 0
    for a in A:
        elt_a += cosine(w,vectors[a])
    for b in B:
        elt_b += cosine(w,vectors[b])
    return 1/len(A)*elt_a - 1/len(B)*elt_b

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score = 0.0
    sum_x=sum_y=0
    for x in X:
        sum_x +=association_strength(vectors[x],A,B,vectors)
    for y in Y:
        sum_y +=association_strength(vectors[y],A,B,vectors)   
    
    score = sum_x - sum_y
    return score


# In[208]:


## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))


# In[ ]:




