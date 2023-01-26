# Fact-aware Language Modeling

This is the Honor Project(COMP4550) for Zhen Wang. The project is supervised by Dr. Omran and Prof. Taylor. The project aims to research the possibility of utilising knowlegdge graph(KG) in language model(LM). All the codes, papers and other resources will be stored in the resources folder.
All the past meetings with Dr. Omran will be noted down in google share doc with the link https://docs.google.com/document/d/18PHuKFnqjeQ0bHksYaKyWbuCZjNEvmRd8yVJ973rPN8/edit.

This repo contains all codes, past meeting records, papers for the honour project.

****** Dataset ******

We constructed two dataset from DBpedia 2014 dump under data folder
* **fact_query_DB**: form text sample by combine query and response. each text has a factual score of either 1 or 0.
* **DBpedia_query**: each row is query and response. Data cleaning has been done to remove text that contains special character. The query is formed by combining subject and predicate. The response is just object.

****** Code ******

RL framework:
- **trl**: folder contains ppo implementation, gpt-2 response and utilities.
- **training_trl**: folder contains three implementation of trl training.
  - threshold-based with classification as response
  - fine-tuning gpt-2 on factual data as factual score with batch reponse
  - linking as factual with batch response
- Finetune on 0,1: folder contains fine-tune gpt-2 with classification on object attempt and fine-tune gpt-2 with 0/1 factual attempt. The trained model are saved in `code/Finetune on 0,1/model/gpt2-text-classifier-model.pt` Please download this folder at https://www.dropbox.com/scl/fo/mhxzt9whk547puohubvhk/h?dl=0&rlkey=2wlwcuvtva750p677dzoaynst

The factual pipeline:
- **REL**: the folder to hold entity linking model that contains code from muli-relation model, ranker, flair ner, ED package.
- **ED model**: the folder contains trained ED model. Please download this folder at https://www.dropbox.com/sh/b0eevkzdock2x5s/AABoluM2W-m0vJspfOSCufQua?dl=0
- **FC**: the folder contains all the code of FC package
- **project_folder**: folder contains wiki 2019 dump and AIDA with other datasets for training and testing. Download Wiki 2019 at http://gem.cs.ru.nl/ed-wiki-2019.tar.gz
- - **`ED_detect.py`**: code to use the trained ED model
- - **`ED_trl.py`**: code to use the trained ED model in trl
- - **`ED_train.py`**: train the ED on AIDA
- - **`highlight.py`**: highlight the result from ED and filtered by LR score
- - **`Spotlight``.py`**: code to use DBpedia spotlight and Opennre by  [Tianyu Gao](https://github.com/gaotianyu1350) .



# Author
Zhen Wang# Fact-aware-Language-Modelling
