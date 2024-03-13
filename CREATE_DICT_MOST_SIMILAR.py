import pandas as pd
import numpy as np
from tqdm import tqdm
import gensim
import gensim.models
from gensim.models import KeyedVectors
import pickle
import time

#input
lang_src = input('insert the source language: ')
lang_trg = input('insert the target language: ')
embedding_src = input('insert the filename of source embedding: ')
embedding_trg = input('insert the filename of target embedding: ')

#output
output_most_similar_dict_src = 'data/most_similar_dictionary_' + lang_src +'.pkl'
output_most_similar_dict_trg = 'data/most_similar_dictionary_' + lang_trg +'.pkl'

# load models
path_save_vec_model_src = 'input/' + embedding_src
path_save_vec_model_trg = 'input/' + embedding_trg

start = time.time()

model_src = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_save_vec_model_src, binary=False,unicode_errors='ignore')
model_trg = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_save_vec_model_trg, binary=False,unicode_errors='ignore')

vocab_src = model_src.vocab
vocab_trg = model_trg.vocab
#vocab_src = model_src.key_to_index
#vocab_trg = model_trg.key_to_index

dict_src = {}
dict_trg = {}

for word in tqdm(vocab_src):
    most_similar_src = model_src.most_similar(word, topn=200)
    dict_src[word] = most_similar_src

#save the dictionary of similar words
open_file = open(output_most_similar_dict_src, "wb")
pickle.dump(dict_src, open_file)
open_file.close()

for word in tqdm(vocab_trg):
    most_similar_trg = model_trg.most_similar(word, topn=200)
    dict_trg[word] = most_similar_trg


end = time.time()
print(f'elapsed time: {end - start}')
#save the dictionary of similar words
open_file = open(output_most_similar_dict_trg, "wb")
pickle.dump(dict_trg, open_file)
open_file.close()