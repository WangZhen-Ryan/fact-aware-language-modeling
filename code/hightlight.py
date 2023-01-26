#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

API_URL = "https://rel.cs.ru.nl/api"
text_doc = "If you're going to try, go all the way - Charles Bukowski"

# Example EL.
el_result = requests.post(API_URL, json={
    "text": text_doc,
    "spans": []
}).json()

# Example ED.
ed_result = requests.post(API_URL, json={
    "text": text_doc,
    "spans": [(41, 16)]
}).json()


# In[4]:


base_url = "C:\\Users\\wz\\Desktop\\REL-main\\project_folder"


# In[3]:


from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner
from REL.server import make_handler

wiki_version = "wiki_2014"


# In[4]:


import os
os.path.abspath(os.getcwd())


# In[5]:


os.path.join(base_url, wiki_version, "generated")


# In[6]:


config = {
    "mode": "eval",
    "model_path": "C:\\Users\\wz\\Desktop\\REL-main\\project_folder\\wiki_2014\\generated\\model",  # or alias, see also tutorial 7: custom models
}

model = EntityDisambiguation(base_url, wiki_version, config)


# In[7]:


# Using Flair:
tagger_ner = load_flair_ner("ner-fast")

# Alternatively, using n-grams:
tagger_ngram = Cmns(base_url, wiki_version, n=5)


# In[8]:


server_address = ("127.0.0.1", 1235)
server = HTTPServer(
    server_address,
    make_handler(
        base_url, wiki_version, model, tagger_ner
    ),
)

try:
    print("Ready for listening.")
    server.serve_forever()
except KeyboardInterrupt:
    exit(0)


# In[ ]:


import requests

IP_ADDRESS = "http://localhost"
PORT = "1235"
text_doc = "If you're going to try, go all the way - Charles Bukowski"

document = {
    "text": text_doc,
    "spans": [],  # in case of ED only, this can also be left out when using the API
}

API_result = requests.post("{}:{}".format(IP_ADDRESS, PORT), json=document).json()


# ## Pipeline integration

# In[5]:


from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner

wiki_version = "wiki_2019"


# In[6]:


def example_preprocessing():
    # user does some stuff, which results in the format below.
    text = "Obama will visit Germany. And have a meeting with Merkel tomorrow."
    #processed = {"test_doc1": [text, [(17,25)]], "test_doc2": [text, []]}
    processed = {"test_doc1": [text, []]}
    return processed

input_text = example_preprocessing()


# In[7]:


mention_detection = MentionDetection(base_url, wiki_version)
tagger_ner = load_flair_ner("ner-fast")
tagger_ngram = Cmns(base_url, wiki_version, n=5)
mentions_dataset, n_mentions = mention_detection.find_mentions(input_text, tagger_ngram)


# In[8]:


config = {
    "mode": "eval",
    "model_path": "C:\\Users\\wz\\Desktop\\REL-main\\project_folder\\wiki_2019\\generated\\model",  # or alias, see also tutorial 7: custom models
}

model = EntityDisambiguation(base_url, wiki_version, config)
predictions, timing = model.predict(mentions_dataset)


# In[9]:


result = process_results(mentions_dataset, predictions, input_text)
result


# ### hightlight package

# In[27]:


for i in result[].values():
    print(i)


# In[ ]:


from functools import reduce
from itertools import chain
from termcolor import colored

text = 'left foot right foot left foot right. Feet in the day, feet at night.'
l1 = ['foot','feet']

print(reduce(lambda t, x: t.replace(*x), chain([text.lower()], ((t, colored(t,'yellow','on_red')) for t in l1)))) 


# In[ ]:





# In[ ]:


import spacy
import re


# In[ ]:


import spacy_dbpedia_spotlight
# load your model as usual
nlp = spacy.load('en_core_web_lg')
# add the pipeline stage
nlp.add_pipe('dbpedia_spotlight')
# get the document
doc = nlp('The president of USA is calling Boris Johnson to decide what to do about coronavirus')
# see the entities
print('Entities', [( ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])
# inspect the raw data from DBpedia spotlight
print(doc.ents[0]._.dbpedia_raw_result)


# In[ ]:


entity_list = [ent.text for ent in doc.ents]
entity_list


# In[ ]:


# find the index of substring(entity) inside the text
text = 'The president of USA is calling Boris Johnson to decide what to do about coronavirus'

entity_pos_dict = {}

for substring in entity_list:
    for match in re.finditer(substring, text):
        index = tuple((match.start(), match.end()))
        entity_pos_dict[substring] = index
print(entity_pos_dict)

model.infer({'text': text,'h': {'pos': list(entity_pos_dict.values())[0]}, 't': {'pos': list(entity_pos_dict.values())[1]}})


# In[ ]:


def random_generate_triple:
    return None


# In[ ]:


triple = ['USA','Boris Johnson', 'head of government']


# In[10]:


# write into .tsv
import csv

with open(r'C:\Users\wz\Desktop\KV-rule-main\code\testing\testingone.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(triple)


# In[11]:


import opennre

model = opennre.get_model('wiki80_cnn_softmax')

model = model.cuda()


# In[12]:


model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).','h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})


# In[ ]:




