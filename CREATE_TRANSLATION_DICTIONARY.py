
#from gensim.models.wrappers import FastText
import pandas as pd
import csv, codecs
from collections import Counter
import numpy as np
from rapidfuzz import fuzz
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm import tqdm
import re
import gensim
import operator
import time
import pickle
#from gensim.models.wrappers import FastText
import string
from googletrans import Translator
from tqdm import tqdm
import time
from re import search

#ft_model = FastText.load_fasttext_format('data_v2/ft_vectors_cbow_100_100_0.1_security.bin')
#input
#lang_src = 'en'
root = input('insert root with data: ')
lang_src = input('insert the source language: ')
#lang_trg = 'it'
lang_trg = input('insert the target language: ')
embedding_src = input('insert the filename of source embedding: ')
embedding_trg = input('insert the filename of target embedding: ')
dictionary_label = lang_src + '_' + lang_trg

#file_italian_embedding = 'input/it.emb.txt' 
#file_english_embedding = 'input/en.emb.txt'

output_translation_dict = root + 'data/translation_dictionary_'+ dictionary_label + '.pkl'

stopwords = root + "utils/stopwords_" + lang_src + ".txt"
stopwords = [line.rstrip('\n') for line in open(stopwords, 'r', encoding='utf-8')]
punct = string.punctuation


# load models
path_save_vec_model_src = root + 'input/' + embedding_src
path_save_vec_model_trg = root + 'input/' + embedding_trg

start = time.time()
model_src = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_save_vec_model_src, binary=False,unicode_errors='ignore')
model_trg = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_save_vec_model_trg, binary=False,unicode_errors='ignore')

stringone = ''
count = 0
index = 0


for i in tqdm(model_src.vocab.keys()):
    if search('', i):
        continue
    else:
        count += 1
        stringone = stringone + i + '\n'
        if count > 300:
            index += 1
            with open(root + "data/" + str(index) + "_file.txt", "w") as output:
                output.write(str(stringone[:-1]))
            stringone = ''
            count = 0
        else:
            continue
if count > 0:
    index += 1
    with open(root + "data/" + str(index) + "_file.txt", "w") as output:
        output.write(str(stringone[:-1]))



for page_number in tqdm(range(1, 10000)):
    try:
        f = open(f'{root}data/{page_number}_file.txt', 'r', encoding="utf-8")
    except:
        break
    
    contents = f.read()
    translator = Translator()
    result = translator.translate(contents, src=lang_src, dest = lang_trg)
    translation = result.text

    try:
        result_test = translator.translate('palla', dest = 'en')
    except:
        print('error')
    if result_test.src != 'it':
        print('si è bloccato, aspetto 1 ora e mezza')
        time.sleep(5400)
        result = translator.translate(contents, src=lang_src, dest = lang_trg)
        translation = result.text

        result_test = translator.translate('palla', dest = 'en')
        print(f'dopo 1 ora e mezza result_test.src dovrebbe essere = it e infatti result_test.src = {result_test.src}')
    try:
        with open(f'{root}data/{page_number}_file_translated.txt', "w", encoding="utf-8") as output:
            output.write(str(translation))
    except UnicodeEncodeError:
        with open(f'{root}data/{page_number}_file_translated.txt', "w", encoding="ascii", errors='ignore') as output:
            output.write(str(translation))

#creo dizionario
list_src_words = []
list_trg_words = []
for page_number in tqdm(range(1,10000)):
    try:
        f = open(f'{root}data/{page_number}_file.txt', 'r', encoding="utf-8")
    except:
        break 
    lines = f.readlines()
    for i in lines:
        list_src_words.append(re.sub('(\\n)$', '', i.lower()))
        
    f_2 = open(f'{root}data/{page_number}_file_translated.txt', 'r', encoding="utf-8")
    lines_2 = f_2.readlines()
    for j in lines_2:
        list_trg_words.append(re.sub('(\\n)$', '', j).lower())
        
end = time.time()
print(f'elapsed time: {end - start}')
        
dict_translation_src_trg = dict(zip(list_src_words, list_trg_words))

open_file = open(output_translation_dict, "wb")
pickle.dump(dict_translation_src_trg, open_file)
open_file.close()
