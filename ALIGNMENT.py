import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
import gensim
import pickle
import collections
import copy
import gensim.models
from gensim.models import KeyedVectors
import embeddings
from cupy_utils import *
import os
import argparse
import sys
import math
import time
import multiprocessing
from multiprocessing import Pool
import statistics


#input
root = input('insert root with data: ')
lang_src = input('insert the source language: ')
lang_trg = input('insert the target language: ')
embedding_src = input('insert the filename of source embedding: ')
embedding_trg = input('insert the filename of target embedding: ')
#limit_similarity = 0.9
#limit_similarity = float(input('insert the maximum limit of similarity for dispersion part: '))
dictionary_label_in = lang_src + '_' + lang_trg
dictionary_label_in_rev = lang_trg + '_' + lang_src
#dictionary_label_out = lang_src + '_' + lang_trg + '_' + str(limit_similarity)
retrieval = input('insert retrieval type: ')
evaluation_precision = input('evaluation_precision @: ')
input_translation_dict_src_trg = root + 'data/translation_dictionary_' + dictionary_label_in +'.pkl'
input_translation_dict_trg_src = root + 'data/translation_dictionary_' + dictionary_label_in_rev +'.pkl'
input_dict_evaluation = root + 'input/' + dictionary_label_in + '.test.txt'

#per velocizzare
input_most_similar_dict_src = root + 'data/most_similar_dictionary_' + lang_src + '.pkl' 
input_most_similar_dict_trg = root + 'data/most_similar_dictionary_' + lang_trg + '.pkl' 

#define folders
data_folder = input('insert data folder: ')
output_folder = input('insert output folder: ')

#output
temp_embedding_src = root + output_folder + '/temp_' + dictionary_label_in + '_src_emb'
temp_embedding_trg = root + output_folder + '/temp_' + dictionary_label_in + '_trg_emb'
path_src_emb_for_alignment = root + output_folder + '/temp_' + dictionary_label_in + '_src_emb_for_alignment.txt'
path_trg_emb_for_alignment = root + output_folder + '/temp_' + dictionary_label_in + '_trg_emb_for_alignment.txt'
output_dict_common_words = root + output_folder + '/dict_common_words_' + dictionary_label_in +'.pkl'
temp_final_output = root + output_folder + '/temp_output_values_' + dictionary_label_in + '.pkl' 
path_trg_word2ind = root + output_folder + '/trg_word2ind_' + dictionary_label_in + '.pkl' 
path_src_word2ind = root + output_folder + '/src_word2ind_' + dictionary_label_in + '.pkl' 
path_translation = root + output_folder + '/translation_' + dictionary_label_in + '.pkl' 
final_output = root + output_folder + '/output_values_' + dictionary_label_in + '.csv' 

# load models
path_save_vec_model_src = root + 'input/' + embedding_src
path_save_vec_model_trg = root + 'input/' + embedding_trg

print('first import dei modelli')
start_time = time.time()

model_src_origin = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_save_vec_model_src, binary=False,unicode_errors='ignore')
model_src_origin.init_sims()
model_src_origin.save(temp_embedding_src)
model_trg_origin = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_save_vec_model_trg, binary=False,unicode_errors='ignore')
model_trg_origin.init_sims()
model_trg_origin.save(temp_embedding_trg)

end_time = time.time()
elapsed_time = end_time - start_time
print('total time for first import dei modelli = ' + str('%.3f'%(elapsed_time)) + " sec \n")

file_to_read_dict_src_trg = open(input_translation_dict_src_trg, "rb")
file_to_read_dict_trg_src = open(input_translation_dict_trg_src, "rb")
dict_translation_src_trg = pickle.load(file_to_read_dict_src_trg)
dict_translation_trg_src = pickle.load(file_to_read_dict_trg_src)
dict_most_frequent_common = {}
vocab_src = model_src_origin.vocab
vocab_trg = model_trg_origin.vocab
#vocab_src = model_src_origin.key_to_index
#vocab_trg = model_trg_origin.key_to_index

#create dictionary of common words among corpora (translating src in trg language)
for i in tqdm(vocab_src):
    try:
        translation = dict_translation_src_trg[i]
    except KeyError:
        continue
    translation_clean = translation.lower()
    if translation_clean in vocab_trg:
        dict_most_frequent_common[i] = translation_clean
    elif i in vocab_trg:
        dict_most_frequent_common[i] = i
    else: 
        continue

#save the dictionary of common words
open_file = open(output_dict_common_words, "wb")
pickle.dump(dict_most_frequent_common, open_file)
open_file.close()

file_to_read_common_words = open(output_dict_common_words, "rb")
dict_common_words = pickle.load(file_to_read_common_words)

#per velocizzare import dictionaries of similar words
file_to_read_dict = open(input_most_similar_dict_src, "rb")
dict_most_similar_src = pickle.load(file_to_read_dict)

file_to_read_dict = open(input_most_similar_dict_trg, "rb")
dict_most_similar_trg = pickle.load(file_to_read_dict)
#define functions

def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
	"""Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
	Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
		(With help from William. Thank you!)
	First, intersect the vocabularies (see `intersection_align_gensim` documentation).
	Then do the alignment on the other_embed model.
	Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
	Return other_embed.
	If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
	"""
	
	# patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
	base_embed.init_sims()
	other_embed.init_sims()

	# make sure vocabulary and indices are aligned
	in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

	# get the embedding matrices
	base_vecs = in_base_embed.syn0norm
	other_vecs = in_other_embed.syn0norm

	# just a matrix dot product with numpy
	m = other_vecs.T.dot(base_vecs) 
	# SVD method from numpy
	u, _, v = np.linalg.svd(m)
	# another matrix operation
	ortho = u.dot(v) 
	# Replace original array with modified one
	# i.e. multiplying the embedding matrix (syn0norm)by "ortho"
	#other_embed.syn0norm = other_embed.syn0 = (other_embed.syn0norm).dot(ortho)
	return ortho

def intersection_align_gensim(m1,m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.vocab.keys())
    vocab_m2 = set(m2.wv.vocab.keys())
    #vocab_m1 = set(m1.key_to_index.keys())
    #vocab_m2 = set(m2.key_to_index.keys())
    # Find the common vocabulary
    common_vocab = vocab_m1&vocab_m2
    if words: common_vocab&=set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,reverse=True)

    # Then for each model...
    for m in [m1,m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.vocab[w].index for w in common_vocab]
        old_arr = m.wv.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.syn0norm = m.wv.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.wv.index2word = common_vocab
        old_vocab = m.wv.vocab
        new_vocab = {}
        for new_index,word in enumerate(common_vocab):
            old_vocab_obj=old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.wv.vocab = new_vocab

    return (m1,m2)

#function for NDCG computation (NDCG is a mesaure that consider both similarity and ranking position of of trg words repsect src words)
def ndcg_score(list_ranks, list_scores):
    ndcg_score = 0
    for score, rank in zip(list_scores, list_ranks):
        add = score/(math.log((rank+2), 2))
        ndcg_score +=add
    return ndcg_score

def imap_bar(func, args, n_processes = (multiprocessing.cpu_count()-1)):
    #print('QUIIIII', args)
    p = Pool(n_processes,maxtasksperchild=5000)
    res_list = []
    with tqdm(total = len(args),mininterval=60) as pbar:
        #args = [list(x) for x in args]
        #for res in tqdm(p.starmap(func, args,chunksize = 10000)):
        for res in tqdm(p.starmap(func, args)):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

def prepare_add_anchors(word_src, word_trg):
    #restricted_word_set = set()
    #vector_src = model_src_origin.wv[word_src]
    vector_src = model_src_origin[word_src]
    #model_src.wv.add('anchor_'+word_trg, vector_src)
    #vector_trg = model_trg_origin.wv[word_trg]
    vector_trg = model_trg_origin[word_trg]
    #model_trg.wv.add('anchor_'+word_trg, vector_trg)
    anchor_name = 'anchor_'+ word_trg
    return (anchor_name, vector_src, vector_trg)

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    where_are_NaNs = np.isnan(m)
    m[where_are_NaNs] = 0
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def create_init_couple_w_ndcg(dict_common_words, dict_translation_src_trg, dict_translation_trg_src, dict_most_similar_src, dict_most_similar_trg, model_trg, model_src):
    for line in tqdm(dict_common_words.items()):
        #from src to trg
        most_similar_src_all = dict_most_similar_src[line[0]]
        most_similar_src = []
        for index, i in enumerate(most_similar_src_all):
            if index >= top_similar:
                break
            else:
                most_similar_src.append(i)
        set_most_similar_src = set()
        dict_rank_most_sim_src = {}

        for index, word_src in enumerate(most_similar_src):
            try:
                word_translated = dict_translation_src_trg[word_src[0]]
            except KeyError:
                continue
            if word_translated == line[1] or word_translated not in vocab_trg:
                continue
            set_most_similar_src.add(word_translated)
            dict_rank_most_sim_src[word_translated] = index

        #select most similar words for each trg anchor
        most_similar_trg_all = dict_most_similar_trg[line[1]]
        most_similar_trg = []
        for index, i in enumerate(most_similar_trg_all):
            if index >= top_similar:
                break
            else:
                most_similar_trg.append(i)
        set_most_similar_trg = set()
        for index_2, word_trg in enumerate(most_similar_trg):
            set_most_similar_trg.add(word_trg[0])
        intersection = set_most_similar_src.intersection(set_most_similar_trg)
        list_sims_for_ndcg_trg = []
        list_indexes_for_ndcg_trg = []

        #compute NDCG for each couple; in particular NDCG is computed cosnidering the similarity between most similar words of src anchor (translated in target language) respect trg anchor and the ranking of those src words
        for i in set_most_similar_src:
        #for i in intersection:
            list_indexes_for_ndcg_trg.append(dict_rank_most_sim_src[i])
            sim_for_ndcg_trg = model_trg.wv.similarity(i, line[1])
            #sim_for_ndcg_trg = model_trg.similarity(i, line[1])
            list_sims_for_ndcg_trg.append(sim_for_ndcg_trg)

        #aggiungo lista rank and scores from trg to src

        #from trg to src
        set_most_similar_trg = set()
        dict_rank_most_sim_trg = {}

        for index, word_trg in enumerate(most_similar_trg):
            try:
                word_translated = dict_translation_trg_src[word_trg[0]]
            except KeyError:
                continue
            if word_translated == line[0] or word_translated not in vocab_src:
                continue
            set_most_similar_trg.add(word_translated)
            dict_rank_most_sim_trg[word_translated] = index

        #select most similar words for each trg anchor
        list_sims_for_ndcg_src = []
        list_indexes_for_ndcg_src = []
        for i in set_most_similar_trg:
        #for i in intersection:
            list_indexes_for_ndcg_src.append(dict_rank_most_sim_trg[i])
            sim_for_ndcg_src = model_src.wv.similarity(i, line[0])
            list_sims_for_ndcg_src.append(sim_for_ndcg_src)
        #aggiunta (calcolo ndcg solo sulle parole dell'intersection)
        #append delle due liste
        list_indexes_for_ndcg = [x for x in list_indexes_for_ndcg_trg]
        for x in list_indexes_for_ndcg_src:
            list_indexes_for_ndcg.append(x)
        list_sims_for_ndcg = [x for x in list_sims_for_ndcg_trg]
        for x in list_sims_for_ndcg_src:
            list_sims_for_ndcg.append(x)
        
        score = ndcg_score(list_indexes_for_ndcg, list_sims_for_ndcg)
        if score > 0:
            dict_anchors[line] = intersection
            dict_anchors_scores[line] = score
        else:
            continue
    return dict_anchors, dict_anchors_scores
    

print('start anchors selection for different parameters: ')

list_output = []
df_output = pd.DataFrame()
#value of similarity to consider for dispersion?
#limit_similarity = median
#list_limit_similarity = [0.6232532262802124, 0.6367987394332886]
#list_limit_similarity = [0.6]
#number of most similar words to consider?
#top_similar = 40
#list_top_similar = [5, 10, 15, 20, 25, 30, 35, 40, 45]
list_top_similar = [35]
#possible maximum value of NDCG allowed for best anchors selection
#list_tollerance_limit = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 ,0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
#list_tollerance_limit = [0.0, 0.03, 0.05, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2, 0.23, 0.25, 0.28, 0.3]
list_tollerance_limit = [0.13]
for top_similar in list_top_similar:
    for tollerance_limit in list_tollerance_limit:  
        print(f'testo alignment con {top_similar} top similar e {tollerance_limit} come tollerance limit')
        dict_anchors = {}
        dict_anchors_scores = {}
        count_it = 0
        count_uk = 0

        #model_src = copy.deepcopy(model_src_origin)
        #model_trg = copy.deepcopy(model_trg_origin)
        model_src = KeyedVectors.load(temp_embedding_src)
        model_trg = KeyedVectors.load(temp_embedding_trg)

        #file_exists_0 = os.path.exists(output_dict_anchors)
        print('start anchors selection')
        start_time = time.time()
        #select most similar words for each couple of common words and traslate the source one in target language 
        #if file_exists_0 == False:
        #select most similar words for each src anchor and translate them
        dict_anchors, dict_anchors_scores = create_init_couple_w_ndcg(dict_common_words, dict_translation_src_trg, dict_translation_trg_src, dict_most_similar_src, dict_most_similar_trg, model_trg, model_src)

        print('Starting lenght of Seed Lexicon:', len(dict_anchors.keys()))

        dict_items = dict_anchors_scores.items()
        list_items = list(dict_items)
        df_anchors = pd.DataFrame(list_items)

        #remove couple with the same translation mantaining the couple with highest NDCG     
        dict_anchors_score_sort = {k: v for k, v in sorted(dict_anchors_scores.items(), key=lambda item: item[1], reverse=True)}
        print('delete anchors with almost one element in common; selecting the couple with highest NDCG')

        column_1 = []
        column_2 = []
        column_3 = []
        for key, values in dict_anchors_score_sort.items():
            column_1.append(key[0])
            column_2.append(key[1])
            column_3.append(values)


        df_anchors_score_sort = pd.DataFrame()
        df_anchors_score_sort['column_1'] = column_1
        df_anchors_score_sort['column_2'] = column_2
        df_anchors_score_sort['column_3'] = column_3

        df_anchors_score_sort_dedup = df_anchors_score_sort.drop_duplicates(subset=['column_1'], keep='first')
        df_anchors_score_sort_dedup = df_anchors_score_sort_dedup.drop_duplicates(subset=['column_2'], keep='first')
        #select just top anchors
        data = df_anchors_score_sort_dedup['column_3']

        #Normalize DCG
        # manual scaling
        minimum = df_anchors_score_sort_dedup['column_3'].min()
        maximum = df_anchors_score_sort_dedup['column_3'].max()
        scaled = [(x - minimum) / (maximum - minimum) for x in data]
        df_anchors_score_sort_dedup['column_3'] = scaled

        #Select best Anchors cutting worsts ones
        tot_anchors_to_mantain = round(len(df_anchors_score_sort_dedup['column_1'])*(1-tollerance_limit))

        df_anchors_score_sort_dedup = df_anchors_score_sort_dedup.head(tot_anchors_to_mantain)

        dict_anchors_score_sort = {}
        for index, row in df_anchors_score_sort_dedup.iterrows():
            tuple_key = (row['column_1'], row['column_2'])
            dict_anchors_score_sort[tuple_key] = row['column_3']

        print(f'after cutting worsts {tollerance_limit} of anchors, anchors become: {len(dict_anchors_score_sort)}')

        list_element_to_delete = []
        for key, values in dict_anchors.items():
            if key not in dict_anchors_score_sort.keys():
                list_element_to_delete.append(key)

        for eliminated in tqdm(list_element_to_delete):
            del dict_anchors[eliminated]

        #Dispersion Part
        #remove anchors too much similar between them, mantainig the couple with highest NDCG
        list_most_similar_values = []
        for key, values in dict_anchors.items():
            most_similar_trg_all = dict_most_similar_trg[key[1]]
            most_similar_trg = []
            for index, i in enumerate(most_similar_trg_all):
                list_most_similar_values.append(i[1])
                break
        
        q3, q1 = np.percentile(list_most_similar_values, [95 ,5])
        list_most_similar_values_clean = []
        for i in list_most_similar_values:
            if i <= q1 or i >= q3:
                continue
            else:
                list_most_similar_values_clean.append(i)

        mean = statistics.mean(list_most_similar_values_clean)
        sd = statistics.stdev(list_most_similar_values_clean)
        #median = statistics.median(list_most_similar_values_clean)
        #limit_similarity = mean
        
        file_temp_dict_anchors = root + data_folder + '/temp_dict_anchors_' + dictionary_label_in + '_' + str(top_similar) + '_' +str(tollerance_limit) + '.pkl'
        open_file = open(file_temp_dict_anchors, "wb")
        pickle.dump(dict_anchors, open_file)
        open_file.close()

        file_temp_dict_anchors_scores = root + data_folder + '/temp_dict_anchors_scores_' + dictionary_label_in + '_' + str(top_similar) + '_' +str(tollerance_limit) + '.pkl'
        open_file = open(file_temp_dict_anchors_scores, "wb")
        pickle.dump(dict_anchors_score_sort, open_file)
        open_file.close()

        #list_limit_similarity = [mean, mean-sd, mean-(2*sd), mean+sd, mean+(2*sd)]
        list_limit_similarity = [mean]
        for limit_similarity in list_limit_similarity:

            print(f'we are going to use the sequent limit similarity: {limit_similarity}')
            output_dict_anchors = root + data_folder + '/dict_anchors_' + str(limit_similarity) + '_' + dictionary_label_in + '_' + str(top_similar) + '_' +str(tollerance_limit) + '.pkl'
            #output_df_anchors = output_folder + '/anchors_' + dictionary_label_in + '_' + str(top_similar) + '_' +str(tollerance_limit) + '.csv'

            #import dict_anchors until now
            file_to_read_temp_dict_anchors = open(file_temp_dict_anchors, "rb")
            dict_anchors = pickle.load(file_to_read_temp_dict_anchors)

            file_to_read_temp_dict_anchors_scores = open(file_temp_dict_anchors_scores, "rb")
            dict_anchors_score_sort = pickle.load(file_to_read_temp_dict_anchors_scores)

            aux_dict_anchors_score_sort = {}
            for key, values in tqdm(dict_anchors_score_sort.items()):
                aux_dict_anchors_score_sort[key[1]] = key

            list_element_to_delete = []

            for key, values in tqdm(dict_anchors_score_sort.items()):
                del aux_dict_anchors_score_sort[key[1]]
                if key in list_element_to_delete:
                    continue
                
                #in order to decrease the computation time we check for each trg word if there are most similar words with similarity higher than the pre fixed limit
                #list_scores = model_trg.most_similar(key[1], topn=100)
                list_scores_all = dict_most_similar_trg[key[1]]
                list_scores = []
                for index, i in enumerate(list_scores_all):
                    if index >= 100000:
                        break
                    else:
                        list_scores.append(i)
                for score in list_scores:
                    if score[1] >= limit_similarity:
                        if score[0] in aux_dict_anchors_score_sort.keys():
                            list_element_to_delete.append(aux_dict_anchors_score_sort[score[0]])
                        else:
                            continue
                    else: 
                        break
                    
            dedup_list_element_to_delete = list(dict.fromkeys(list_element_to_delete))
            print(f'after the dispersion the number of anchors become: {len(dict_anchors.keys())-len(dedup_list_element_to_delete)}')

            for eliminated in tqdm(dedup_list_element_to_delete):
                del dict_anchors[eliminated]

            open_file = open(output_dict_anchors, "wb")
            pickle.dump(dict_anchors, open_file)
            open_file.close()

            dict_items = dict_anchors.items()
            list_items = list(dict_items)
            #df_anchors_new = pd.DataFrame(list_items)
            #df_anchors_new.to_csv(output_df_anchors)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('total time for anchors selection = ' + str('%.3f'%(elapsed_time)) + " sec \n")
            #ALIGNMENT
            print('Perform alignment using the semantic Seed Lexicon')
            input_dict_anchors = root + data_folder + '/dict_anchors_' + str(limit_similarity) + '_' + dictionary_label_in + '_' + str(top_similar) + '_' +str(tollerance_limit) + '.pkl'
            path_model_prjected = root + output_folder + '/' + lang_src + '.emb_projected_' + str(limit_similarity) + '_' + str(top_similar) + '_' + str(tollerance_limit) + '.txt'
            path_model_not_prjected = root + output_folder + '/' + lang_trg + '.emb_not_projected.txt'
            #path_model_prjected = output_folder + '/' + lang_trg + '.emb_projected_' + str(limit_similarity) + '_' + str(top_similar) + '_' + str(tollerance_limit) + '.txt'
            #path_model_not_prjected = output_folder + '/' + lang_src + '.emb_not_projected.txt'
            

            model_src = KeyedVectors.load(temp_embedding_src)
            model_trg = KeyedVectors.load(temp_embedding_trg)
            src_words = model_src.vocab
            trg_words = model_trg.vocab
            #src_words = model_src.key_to_index
            #trg_words = model_trg.key_to_index

            # Build word to index map
            src_word2ind = {word: i for i, word in enumerate(src_words)}
            trg_word2ind = {word: i for i, word in enumerate(trg_words)}

            file_to_read_anchors = open(input_dict_anchors, "rb")
            dict_anchors = pickle.load(file_to_read_anchors)

            print('start 0 part of alignment (inception)')
            start_time = time.time()

            tuple_keys = dict_anchors.keys()
            list_anchors_to_add = imap_bar(prepare_add_anchors, tuple_keys)
            temp_f = open(f'{path_save_vec_model_src}', encoding='utf-8', errors='surrogateescape')
            for line in temp_f:
                nothing, second_col = line.split()
                break

            first_col = str(len(list_anchors_to_add) - 1) 

            with open(path_src_emb_for_alignment, 'w', encoding='utf-8') as out:
                out.write(first_col + ' ' + second_col)
                out.write('\n')
                for i in list_anchors_to_add:
                    out.write(i[0] + ' ' + ' '.join(['%.9g' % x for x in i[1]]))  
                    out.write('\n')
            out.close()

            with open(path_trg_emb_for_alignment, 'w', encoding='utf-8') as out:
                out.write(first_col + ' ' + second_col)
                out.write('\n')
                for i in list_anchors_to_add:
                    out.write(i[0] + ' ' + ' '.join(['%.9g' % x for x in i[2]]))  
                    out.write('\n')
            out.close()

            model_src_for_alignment = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_src_emb_for_alignment, binary=False,unicode_errors='ignore')
            model_trg_for_alignment = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(path_trg_emb_for_alignment, binary=False,unicode_errors='ignore')

            ortho = smart_procrustes_align_gensim(model_trg_for_alignment, model_src_for_alignment, words=None)
            #ortho = smart_procrustes_align_gensim(model_src_for_alignment, model_trg_for_alignment, words=None)

            model_src = KeyedVectors.load(temp_embedding_src)
            model_trg = KeyedVectors.load(temp_embedding_trg)
            #fine check
            model_projected_complete = model_src
            model_not_projected_complete = model_trg
            #model_projected_complete = model_trg
            #model_not_projected_complete = model_src

            #project src embedding in trg space
            #model_projected_complete = copy.deepcopy(model_projected_complete_copy)
            model_projected_complete.init_sims()
            model_projected_complete.wv.syn0norm = model_projected_complete.wv.syn0 = (model_projected_complete.wv.syn0norm).dot(ortho)

            temp_f = open(f'{path_save_vec_model_src}', encoding='utf-8', errors='surrogateescape')
            for line in temp_f:
                first_col, second_col = line.split()
                break

            m = np.asarray(model_projected_complete.wv.vectors)
            words = model_projected_complete.wv.index2entity
            with open(path_model_prjected, 'w', encoding='utf-8') as out:
                out.write(first_col + ' ' + second_col)
                out.write('\n')
                for i in range(len(words)):
                    out.write(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]))
                    out.write('\n')
                    
            out.close()

            temp_f = open(f'{path_save_vec_model_trg}', encoding='utf-8', errors='surrogateescape')
            for line in temp_f:
                first_col, second_col = line.split()
                break

            m = np.asarray(model_not_projected_complete.wv.vectors)
            words = model_not_projected_complete.wv.index2entity
            with open(path_model_not_prjected, 'w', encoding='utf-8') as out:
                out.write(first_col + ' ' + second_col)
                out.write('\n')
                for i in range(len(words)):
                    out.write(words[i] + ' ' + ' '.join(['%.6g' % x for x in m[i]]))
                    out.write('\n')
                    
            out.close()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('total time for 3 part of alignment (projection) = ' + str('%.3f'%(elapsed_time)) + " sec \n")
            #Embedding Alignemnt EVALUATION
            #This part is taken from gitHub: 
            print('start evaluation')
            start_time = time.time()
            BATCH_SIZE = 500
            # Parse command line arguments
            parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
            parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
            parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
            parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
            parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
            parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
            parser.add_argument('--seed', type=int, default=0, help='the random seed')
            parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
            parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
            args = parser.parse_args()

            # Choose the right dtype for the desired precision
            if args.precision == 'fp16':
                dtype = 'float16'
            elif args.precision == 'fp32':
                dtype = 'float32'
            elif args.precision == 'fp64':
                dtype = 'float64'

            # Read input embeddings
            #srcfile = open(path_model_not_prjected, encoding=args.encoding, errors='surrogateescape')
            #trgfile = open(path_model_prjected, encoding=args.encoding, errors='surrogateescape')
            srcfile = open(path_model_prjected, encoding=args.encoding, errors='surrogateescape')
            trgfile = open(path_model_not_prjected, encoding=args.encoding, errors='surrogateescape')
            src_words, x = embeddings.read(srcfile, dtype=dtype)
            trg_words, z = embeddings.read(trgfile, dtype=dtype)
            
            # NumPy/CuPy management
            if args.cuda:
                if not supports_cupy():
                    print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
                    sys.exit(-1)
                xp = get_cupy()
                x = xp.asarray(x)
                z = xp.asarray(z)
            else:
                xp = np
            xp.random.seed(args.seed)

            # Length normalize embeddings so their dot product effectively computes the cosine similarity
            if not args.dot:
                embeddings.length_normalize(x)
                embeddings.length_normalize(z)

            # Build word to index map
            src_word2ind = {word: i for i, word in enumerate(src_words)}
            trg_word2ind = {word: i for i, word in enumerate(trg_words)}

            open_file = open(path_src_word2ind, "wb")
            pickle.dump(src_word2ind, open_file)
            open_file.close()

            open_file = open(path_trg_word2ind, "wb")
            pickle.dump(trg_word2ind, open_file)
            open_file.close()


            # Read dictionary and compute coverage
            f = open(input_dict_evaluation, encoding=args.encoding, errors='surrogateescape')
            src2trg = collections.defaultdict(set)
            oov = set()
            vocab = set()
            for line in f:
                src, trg = line.split()
                try:
                    src_ind = src_word2ind[src]
                    trg_ind = trg_word2ind[trg]
                    src2trg[src_ind].add(trg_ind)
                    vocab.add(src)
                except KeyError:
                    oov.add(src)
            src = list(src2trg.keys())
            oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
            coverage = len(src2trg) / (len(src2trg) + len(oov))

            # Find translations
            translation = collections.defaultdict(int)
            if retrieval == 'nn':  # Standard nearest neighbor
                for i in range(0, len(src), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(src))
                    similarities = x[src[i:j]].dot(z.T)
                    nn = similarities.argmax(axis=1).tolist()
                    for k in range(j-i):
                        translation[src[i+k]] = nn[k]
            elif retrieval == 'invnn':  # Inverted nearest neighbor
                best_rank = np.full(len(src), x.shape[0], dtype=int)
                best_sim = np.full(len(src), -100, dtype=dtype)
                for i in range(0, z.shape[0], BATCH_SIZE):
                    j = min(i + BATCH_SIZE, z.shape[0])
                    similarities = z[i:j].dot(x.T)
                    ind = (-similarities).argsort(axis=1)
                    ranks = asnumpy(ind.argsort(axis=1)[:, src])
                    sims = asnumpy(similarities[:, src])
                    for k in range(i, j):
                        for l in range(len(src)):
                            rank = ranks[k-i, l]
                            sim = sims[k-i, l]
                            if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                                best_rank[l] = rank
                                best_sim[l] = sim
                                translation[src[l]] = k
            elif retrieval == 'invsoftmax':  # Inverted softmax
                sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
                partition = xp.zeros(z.shape[0])
                for i in range(0, len(sample), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(sample))
                    partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
                for i in range(0, len(src), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(src))
                    p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
                    nn = p.argmax(axis=1).tolist()
                    for k in range(j-i):
                        translation[src[i+k]] = nn[k]
            elif retrieval == 'csls':  # Cross-domain similarity local scaling
                knn_sim_bwd = xp.zeros(z.shape[0])
                for i in range(0, z.shape[0], BATCH_SIZE):
                    j = min(i + BATCH_SIZE, z.shape[0])
                    knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
                for i in range(0, len(src), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(src))
                    similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
                    #nn = similarities.argmax(axis=1).tolist()
                    #aggiunta
                    nn_new = np.argpartition(similarities, int(evaluation_precision), axis=1)[:, int(evaluation_precision):].tolist()
                    #fine aggiunta
                    for k in range(j-i):
                        #translation[src[i+k]] = nn[k]
                        #aggiunta
                        translation[src[i+k]] = nn_new[k]
                        #fine aggiunta

            open_file = open(path_translation, "wb")
            pickle.dump(translation, open_file)
            open_file.close()
            # Compute accuracy
            #accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])
            #aggiunta
            accuracy = np.mean([1 if set(translation[i]) & set(src2trg[i]) else 0 for i in src])
            #fine aggiunta
            print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))
            list_output.append((limit_similarity, top_similar, tollerance_limit ,len(dict_anchors), accuracy))
            os.remove(path_model_not_prjected)
            os.remove(path_model_prjected)


            open_file = open(temp_final_output, "wb")
            pickle.dump(list_output, open_file)
            open_file.close()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('total time for evaluation = ' + str('%.3f'%(elapsed_time)) + " sec \n")
        os.remove(file_temp_dict_anchors)
        


list_1_column = []
list_2_column = []
list_3_column = []
list_4_column = []
list_5_column = []

for i in list_output:
    list_1_column.append(i[0])
    list_2_column.append(i[1])
    list_3_column.append(i[2])
    list_4_column.append(i[3])
    list_5_column.append(i[4])

df_output['limit similarity for dispersion'] = list_1_column
df_output['top similar'] = list_2_column
df_output['tollerance'] = list_3_column
df_output['tot anchors'] = list_4_column
df_output['accuracy'] = list_5_column

df_output.to_csv(final_output)