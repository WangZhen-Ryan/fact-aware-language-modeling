#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


# In[38]:


# build confidence score from Logistic Regression
datasets = TrainingEvaluationDatasets(base_url, wiki_version).load()


# In[39]:


config = {
    "mode": "train",
    "model_path": "{}/{}/generated/model".format(
        base_url, wiki_version
    ),
}
model = EntityDisambiguation(base_url, wiki_version, config)


# In[40]:


# train
if config["mode"] == "train":
    model.train(
        datasets["aida_train"], {k: v for k, v in datasets.items() if k != "aida_train"}
    )
else:
    model.evaluate({k: v for k, v in datasets.items() if "train" not in k})


# In[42]:


# train LR model

model_path_lr = "{}/{}/generated/".format(base_url, wiki_version)

model.train_LR(
    datasets,
    model_path_lr
)

