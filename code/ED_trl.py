#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.training_datasets import TrainingEvaluationDatasets
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

from functools import reduce
from itertools import chain
from termcolor import colored

base_url = "C:\\Users\\wz\\Desktop\\REL-main\\project_folder"
working_url = os.path.abspath(os.getcwd())
wiki_version = "wiki_2019"
wiki_path = os.path.join(base_url, wiki_version, "generated")


# In[2]:


# example input text
text1 = 'Obama will visit Germany. And have a meeting with Merkel tomorrow.'
text2 = 'The president of USA is calling Boris Johnson'
text3 =  '''Trump lost the 2020 presidential election to Joe Biden but refused to concede defeat,
falsely claiming widespread electoral fraud and attempting to overturn the results by pressuring government officials, 
mounting scores of unsuccessful legal challenges, and obstructing the presidential transition. On January 6, 2021, 
Trump urged his supporters to march to the Capitol, which many of them then attacked, 
resulting in multiple deaths and interrupting the electoral vote count.'''


# In[4]:


def EL_processing(text):
    processed = {"test_doc1": [text, []]}
    return processed


# In[5]:


mention_detection = MentionDetection(base_url, wiki_version)
tagger_ner = load_flair_ner("ner-fast")
tagger_ngram = Cmns(base_url, wiki_version, n=5)


# In[21]:


# factual score with found entities fractional 
def factual(text):
    text_p =  EL_processing(text)
    mentions_dataset, n_mentions = mention_detection.find_mentions(text_p, tagger_ngram)
    config = {
    "mode": "eval",
    "model_path": "C:\\Users\\wz\\Desktop\\REL-main\\project_folder\\wiki_2019\\generated\\model",  
    }

    model = EntityDisambiguation(base_url, wiki_version, config)
    predictions, timing = model.predict(mentions_dataset)
    
    result = process_results(mentions_dataset, predictions, EL_processing(text))
    # to prevent the case where no entity is linked
    if len(result) == 0:
        return 0

    # trim LR score less than e-77
    result = {val for val in list(result.values())[0] if val[4] > 10**-77}
    
    #print(result)
    
    found_words=[]
    for tu in result:
        s_index = tu[0]
        e_index =  tu[1]
        word = tu[2]
        found_words.append(word)
        
    factual = len(found_words)/len(text.split())*10
    
    # print out found words with entities
#     print("Linking result: ")
#     for i in (result):
#         print(i[2], u"\u2192", i[3], " with a score of ",i[4]) 
    
    return factual


# In[22]:


# factual(text3)


# In[ ]:




