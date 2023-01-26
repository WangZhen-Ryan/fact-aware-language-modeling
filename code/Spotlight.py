#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
import re
import spacy_dbpedia_spotlight

import opennre

from functools import reduce
from itertools import chain
from more_itertools import pairwise
from termcolor import colored

import itertools


# In[2]:


# load your model as usual
nlp = spacy.load('en_core_web_lg')
# add the pipeline stage
nlp.add_pipe('dbpedia_spotlight')

# open NRE
model = opennre.get_model('wiki80_cnn_softmax')
model = model.cuda()

# example input text
text = "Donald John Trump used to be president of USA"
text3 =  '''Trump lost the 2020 presidential election to Joe Biden but refused to concede defeat,
falsely claiming widespread electoral fraud and attempting to overturn the results by pressuring government officials, 
mounting scores of unsuccessful legal challenges, and obstructing the presidential transition. On January 6, 2021, 
Trump urged his supporters to march to the Capitol, which many of them then attacked, 
resulting in multiple deaths and interrupting the electoral vote count.'''
short_text = 'The president of USA is calling Boris Johnson to decide what to do about coronavirus'
long_text = '''Coronaviruses are a group of related RNA viruses that cause diseases in mammals and birds. 
In humans and birds, they cause respiratory tract infections that can range from mild to lethal.
Mild illnesses in humans include some cases of the common cold (which is also caused by other viruses, 
predominantly rhinoviruses), while more lethal varieties can cause SARS, MERS and COVID-19, 
which is causing the ongoing pandemic. In cows and pigs they cause diarrhea, 
while in mice they cause hepatitis and encephalomyelitis.'''
long_text2 = ''' He implemented a controversial family separation policy for migrants apprehended at the U.S.–Mexico border. 
Trump's demand for the federal funding of a border wall resulted in the longest US government shutdown in history. 
He deployed federal law enforcement forces in response to the racial unrest in 2020. Trump's "America First" 
foreign policy was characterized by unilateral actions, disregarding traditional allies.
The administration implemented a major arms sale to Saudi Arabia; denied citizens from several Muslim-majority countries entry into the U.S; 
recognized Jerusalem as the capital of Israel; and brokered the Abraham Accords,a series of normalization agreements between Israel and
various Arab states. His administration withdrew U.S. troops from northern Syria, allowing Turkey to occupy the area. 
His administration also made a conditional deal with the Taliban to withdraw U.S. troops from Afghanistan in 2021. 
Trump met North Korea's leader Kim Jong-un three times. Trump withdrew the U.S. from the Iran nuclear agreement and later escalated tensions 
in the Persian Gulf by ordering the assassination of General Qasem Soleimani. Robert Mueller's Special Counsel investigation (2017–2019) 
concluded that Russia interfered to favor Trump's candidacy and that while the prevailing evidence "did not establish that members of 
the Trump campaign conspired or coordinated with the Russian government",
possible obstructions of justice occurred during the course of that investigation.'''
def highight(text):
    # returns: find the index of substring(entity) inside the text
    # like {'USA': (17, 20), 'Boris Johnson': (32, 45), 'coronavirus': (73, 84)}
    
    # get the doc
    doc = nlp(text)
    # print the entities with corresponding entity in DBpedia
    # print('Entities', [( ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])
    # print raw data from DBpedia spotlight with similarity score
    # print(doc.ents[0]._.dbpedia_raw_result)
    entity_list = [ent.text for ent in doc.ents]
    # track entity with index 
    entity_pos_dict = {}
    for substring in entity_list:
        for match in re.finditer(substring, text):
            index = tuple((match.start(), match.end()))
            entity_pos_dict[substring] = index
            
    # print out found words with entity
    # print(reduce(lambda t, x: t.replace(*x), chain([text], ((t, colored(t,'yellow','on_red')) for t in entity_list)))) 
    return entity_pos_dict


# In[3]:


# entity_pos_dict = highight(long_text)
# entity_pos_dict.values()


# In[4]:


# single pair entity infer RE
def inferRE(text,entity_pos_dict,start_index, end_index):
    RE = model.infer({'text': text,'h': {'pos': (start_index)}, 't': {'pos': (end_index)}})
    return RE


# In[5]:


# return unique combination
# [a,b,c] -> [(a,b),(a,c)...] unique combinations of length 2
# `elements` does not contain duplicates.
def unique_combinations(elements: list[tuple]) -> list[tuple[tuple, tuple]]:
    return list(itertools.combinations(elements, 2))


# In[6]:


# for i in unique_combinations(entity_pos_dict.values()):
#     print(inferRE(long_text,entity_pos_dict, (i[0]),(i[1])))


# In[7]:


#print([p for p, _ in zip(pairwise(entity_pos_dict.values()), range(len(entity_pos_dict.values())))])


# In[8]:


# for i in [p for p, _ in zip(pairwise(entity_pos_dict.values()), range(len(entity_pos_dict.values())))]:
#     print(inferRE(long_text,entity_pos_dict, (i[0]),(i[1])))


# In[9]:


# random match any triples
# return top k triple with RE
def filter_RE(text,k):
    
    entity_pos_dict = highight(text)
    
    REs = []
    # all possible combination of two entities
    all_possible_combination = unique_combinations(entity_pos_dict.values())
    # trimed so the RE can only takes on closed pair entities
    trimed = [p for p, _ in zip(pairwise(entity_pos_dict.values()), range(len(entity_pos_dict.values())))]
    for combination in trimed:
        RE = model.infer({'text': text,'h': {'pos': list(combination[0])}, 't': {'pos': list(combination[1])}})
        REs.append(RE)
        
    # get score
    REs_score = [i[1] for i in REs]
    
    # *remove duplicate
    REs_score_d = list(dict.fromkeys(REs_score))
    
    # get top k score and its relation
    numbers = REs_score_d
    numbers_copy = REs_score
    
    if len(REs_score) == 0: # no re found
        return None
    elif len(REs_score) <= k: # too big k
        # print out found re with entities
        re_entity_list = []
        count = 0
        for i in trimed:
            e1 = text[i[0][0]:i[0][1] ]
            e2 = text [i[1][0]:i[1][1]]

            re_entity_list.append(e1)
            re_entity_list.append(e2)
            print(e1, u"\u2192", e2, "has a relation of ", colored(REs[count][0],'white','on_blue'), " with score of ", REs[count][1]) 
            count += 1
        print(reduce(lambda t, x: t.replace(*x), chain([text], ((t, colored(t,'red','on_yellow')) for t in re_entity_list)))) 

        return REs
    
    top = list() # List to store the greatest values
    top_index = list()# List to store the greatest value's index

    for i in range(0, k): 
        max = numbers[0] 
        for j in numbers: 
            if j > max: 
                max = j 
        top.append(max) # Add the gratest to the top list
        top_index.append(numbers_copy.index(max))
        numbers.remove(max) # Now remove the greatest so we can proceed to find the next greatest
    
#     print(top_index)
#     print(top)
    top_k_RE = [REs[index] for index in top_index]
    top_k_entity = [all_possible_combination[index] for index in top_index]
    
    # print out found re with entities
    re_entity_list = []
    for i in range(k):
        e1 = text[(top_k_entity[i][0])[0]:(top_k_entity[i][0])[1]]
        e2 = text [(top_k_entity[i][1])[0]:(top_k_entity[i][1])[1]]
        
        re_entity_list.append(e1)
        re_entity_list.append(e2)
        print(e1, u"\u2192", e2, "has a relation of ", colored(top_k_RE[i][0],'white','on_blue'), " with score of ", top_k_RE[i][1]) 
    
    print(reduce(lambda t, x: t.replace(*x), chain([text], ((t, colored(t,'red','on_yellow')) for t in re_entity_list)))) 
    
    return top_k_RE,top_k_entity


# In[10]:


filter_RE(text3,k=3)


# In[ ]:





# In[ ]:




