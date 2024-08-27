#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install gensim')


# In[2]:


get_ipython().system('pip install python-Levenshtein')


# In[4]:


import gensim
import pandas as pd


# Reading and Exploring the Dataset
# The dataset we are using here is a subset of Amazon reviews from the Cell Phones & Accessories category. The data is stored as a JSON file and can be read using pandas.
# 
# Link to the Dataset: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz

# In[7]:


df = pd.read_json("reviews_Cell_Phones_and_Accessories_5.json", lines=True)
df


# In[8]:


df.shape


# In[9]:


df.reviewText[0]


# Simple Preprocessing & Tokenization
# The first thing to do for any data science task is to clean the data. For NLP, we apply various processing like converting all the words to lower case, trimming spaces, removing punctuations. This is something we will do over here too.
# 
# Additionally, we can also remove stop words like 'and', 'or', 'is', 'the', 'a', 'an' and convert words to their root forms like 'running' to 'run'

# In[10]:


gensim.utils.simple_preprocess("They look good and stick good! I just don't like the rounded shape because I was always bumping it and Siri kept popping up and it was irritating. I just won't buy a product like this again")


# In[11]:


review_text = df.reviewText.apply(gensim.utils.simple_preprocess)


# In[12]:


review_text


# In[13]:


review_text.loc[0]


# #### Training the Word2Vec Model
# Train the model for reviews. Use a window of size 10 i.e. 10 words before the present word and 10 words ahead. A sentence with at least 2 words should only be considered, configure this using min_count parameter.
# 
# Workers define how many CPU threads to be used.

# In[14]:


model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4,
)


# #### Build Vocabulary

# In[15]:


model.build_vocab(review_text, progress_per=1000)


# #### Train the Word2Vec Model

# In[16]:


model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)


# #### Save the Model

# In[17]:


model.save("./word2vec-amazon-cell-accessories-reviews-short.model")


# #### Finding Similar Words and Similarity between words

# In[18]:


model.wv.most_similar("bad")


# In[19]:


model.wv.similarity(w1="cheap", w2="inexpensive")


# In[20]:


model.wv.similarity(w1="great", w2="good")


# In[ ]:




