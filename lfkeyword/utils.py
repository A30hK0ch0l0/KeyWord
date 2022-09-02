# -*- coding: utf-8 -*-
#  using normalizer hazm

# +
#import time
#start  = time.time()

import re
import gensim
import math
import ast
import pandas as pd
import numpy as np
import hazm
#from hazm                   import *
#from hazm                   import POSTagger
#from hazm.Chunker           import tree2brackets
#from normalizer             import Normalizer
#from gensim.models.word2vec import Word2Vec
from itertools               import compress
import os

#end  = time.time()
#print('Imports                 in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))


# +
# start  = time.time()

data_path = f'{os.path.dirname(os.path.abspath(__file__))}/data'

w2v_model = gensim.models.Word2Vec.load(f'{data_path}/w2v/w2v.model')
# data = pd.read_csv("/home/saba/keywordsummerization_git_persian/input_data/data_random_10000.csv")
stop_word = pd.read_csv(f'{data_path}/all_stopwords.csv')
stop_word.dropna(inplace=True)
# data['ner_list'] = data['ner_list'].apply(ast.literal_eval)
norm      = hazm.Normalizer()

tagger    = hazm.POSTagger(model = f'{data_path}/model_tagger/postagger.model')
chunker   = hazm.Chunker(  model = f'{data_path}/model_tagger/chunker.model'  )

# end  = time.time()
# print('Read Model & Stop Words in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))

# +
# start = time.time()

stop_word['st_normal'] = stop_word['word'].apply(norm.normalize)
set_stopwords          = set(stop_word['st_normal'])
list_stopwords         = list(set_stopwords)

# end = time.time()
# print('Normalize Stop Words    in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
# -

def regex(text):
    
    # start  = time.time()
    
    # Removing emails
    text = re.sub(r'[\w\.\-]+@[\w\.\-]+\.[\w\-]+', '', text)
    # Removing urls
    text = re.sub(r'((?:(?:https?)://)?(?:www.)?\w[\w\-\.]*\.[a-z]{2,}[\w\.\-\/#]*(?:\?(?:\w+=\w+&?)+)?)', '', text)
    # Removing mentions
    text = re.sub(r'@[a-z\d_]+', '', text)
    # removing rt
    text = re.sub(r"RT", '', text)
    # Removing extra space
    text = re.sub(r'\s+', ' ', text)
    # Remove enter and tab
    text = re.sub(r'[\n|\r|\t]', ' ', text)
    text = re.sub("[.]", " .", text)
    text = re.sub("[،]", " ،", text)
    text_regexed = text.strip()
    
    # end  = time.time()
    # print('Regex                   in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return text_regexed

def remove_emojies(text):
    
    # start  = time.time()
    
    emoj1 = re.compile(
        r'\u274B|\uFD3E|\u00A9|\u00AE|\u203C|\u2049|\u2122|\u2139|\u2194|\u2195|\u2196|\u2197|\u2198|\u2199|\u21A9|\u21AA|\u231A|\u231B|\u2328|\u23CF|\u23E9|\u23EA|\u23EB|\u23EC|\u23ED|\u23EE|\u23EF|\u23F0|\u23F1|\u23F2|\u23F3|\u23F8|\u23F9|\u23FA|\u24C2|\u25AA|\u25AB|\u25B6|\u25C0|\u25FB|\u25FC|\u25FD|\u25FE|\u2600|\u2601|\u2602|\u2603|\u2604|\u260E|\u2611|\u2614|\u2615|\u2618|\u261D|\u2620|\u2622|\u2623|\u2626|\u262A|\u262E|\u262F|\u2638|\u2639|\u263A|\u2640|\u2642|\u2648|\u2649|\u264A|\u264B|\u264C|\u264D|\u264E|\u264F|\u2650|\u2651|\u2652|\u2653|\u265F|\u2660|\u2663|\u2665|\u2666|\u2668|\u267B|\u267E|\u267F|\u2692|\u2693|\u2694|\u2695|\u2696|\u2697|\u2699|\u269B|\u269C|\u26A0|\u26A1|\u26A7|\u26AA|\u26AB|\u26B0|\u26B1|\u26BD|\u26BE|\u26C4|\u26C5|\u26C8|\u26CE|\u26CF|\u26D1|\u26D3|\u26D4|\u26E9|\u26EA|\u26F0|\u26F1|\u26F2|\u26F3|\u26F4|\u26F5|\u26F7|\u26F8|\u26F9|\u26FA|\u26FD|\u2702|\u2705|\u2708|\u2709|\u270A|\u270B|\u270C|\u270D|\u270F|\u2712|\u2714|\u2716|\u271D|\u2721|\u2728|\u2733|\u2734|\u2744|\u2747|\u274C|\u274E|\u2753|\u2754|\u2755|\u2757|\u2763|\u2764|\u2795|\u2796|\u2797|\u27A1|\u27B0|\u27BF|\u2934|\u2935|\u2B05|\u2B06|\u2B07|\u2B1B|\u2B1C|\u2B50|\u2B55|\u3030|\u303D|\u3297|\u3299|\u2765|\u2710|\u270E|\u2666|\u2664|\u2662|\u2661|\u261D|\u261C|\u261E|\u261F|\u2611|\u2612|\u2610')
    emoj2 = re.compile(r'\#️⃣|\*️⃣|0️⃣|1️⃣|2️⃣|3️⃣|4️⃣|5️⃣|6️⃣|7️⃣|8️⃣|9️⃣')
    emoj3 = re.compile(r'[\U0001F000-\U0001FADF]')

    text = re.sub(emoj1 , '' , text)
    text = re.sub(emoj2 , '' , text)
    text = re.sub(emoj3 , '' , text)

    text = re.sub(r'\s+', ' ', text)

    # end  = time.time()
    # print('Remove Emojies          in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return text


def fix_hashtags(text: str):
    
    # start  = time.time()  

    text = re.sub('#', ' #', text)
    text = re.sub('# ', '#', text)

    # Remote last hashtags
    text_words = text.split()
    while (len(text_words) and text_words[-1].startswith('#')):
        text_words.pop()
    if 0 == len(text_words):
        return   ''
    text       = ' '.join(text_words)
    text_fixed = text.strip()

    # end  = time.time()
    # print('Fix HashTags            in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return text_fixed


# +
def get_n_gram_words(text):
    
    # start = time.time()
    
    n_gram_finded   = []
    stop_word_types = ['PP', 'VP']
    #tagger  = POSTagger(model='keyword_extraction/model_tagger/postagger.model')
    #chunker = Chunker(  model='keyword_extraction/model_tagger/chunker.model'  )
    tagged = tagger.tag(hazm.word_tokenize(text     ))
    result = hazm.tree2brackets(chunker.parse(tagged))
    words  = re.findall('\[(.*?)(\w+)\]', result     )
    for word in words:
        word_type  = word[1].strip()
        word_value = word[0].strip()
        if ((word_type not in stop_word_types) and (word_value not in (n_gram_finded + list_stopwords))):
            n_gram_finded.append(word_value)

#    n_gram_phrase = []
#    for phrase in n_gram_finded:
#        if phrase not in list_stopwords:
#            n_gram_phrase.append(phrase)
    
    n_gram_phrase = n_gram_finded
        
    # end = time.time()
    # print('Get N Gram Words        in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return n_gram_phrase


# -

def word_count(str):
    
    # start = time.time()
    
    #counts = dict()
    counts  = {'': 1}
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word]  = 1

    total_words = sum(counts.values())
    for key, value in counts.items():
        probability  = value / total_words
        #counts[key] = round(probability, 2)
        counts[key]  = probability
    
    # end = time.time()
    # print('Word Count              in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return counts

def convertor_dic_for_value(dic):
    
    # start = time.time()
    
    data        = list(dic.items())
    an_array    = np.array(data)
    array_value = an_array[:, 1]
    brop_matrix = array_value.astype(float)
    brop_matrix = np.transpose(brop_matrix)
    
    # end = time.time()
    # print('Convertor Dic For Value in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return brop_matrix

# +
def normalizer_function(x):
    
#   start = time.time()
    
    x_normalized = x/np.linalg.norm(x)
    
#   end = time.time()
#   print('Normalizer Function     in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return x_normalized


# -

def matrix_embeding_words(lst):
    
    # start  = time.time()
    
    word_embeding = {}
    for word in lst:
        try:
            word_embeding[word] = w2v_model[word]
        except:
            word_embeding[word] = np.zeros((100,))

    emdedding_size   = 100
    embedding_matrix = np.zeros((len(word_embeding), emdedding_size))

    for index, (word, value) in enumerate(word_embeding.items()):
        embedding_vector = word_embeding.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    
    vv                  = np.apply_along_axis(normalizer_function, 1, embedding_matrix)
    #where_are_NaNs     = np.isnan(vv)
    #vv[where_are_NaNs] = 0
    np.nan_to_num(vv, copy = False)
    
    # end = time.time()
    # print('Matrix Embeding Words   in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return vv

# D=(sum(A.AT))/unique words  distance matrix
def distance_matrix(vv):
    
    # start = time.time()
    
    count_words_eaxh_text = len(vv)
    b  = np.transpose(vv)
    a  = np.matmul(vv, b)
    DD = np.sum(a, axis = 1)  # قطر اصلی یک
    D  = DD / count_words_eaxh_text
    #D  = np.around(D, 2)
    
    # end = time.time()
    # print('Distance Matrix         in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return D

# SCORE = B*D
def score_matrix(brop_matrix, D):
    
    # start = time.time()
    
    f = np.log2(brop_matrix)
    multiply = f * D
    #score = np.round(multiply, 2)
    score = multiply
    score = np.absolute(score)
    
    # end = time.time()
    # print('Score Matrix            in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return score

# def min_score(array):
#     max_number_top_word = 15
#     if len(array) <= 4:
#         list_of_indices_min_value = list(np.arange(len(array)))
#     elif 4<len(array)<=50:
#         # number_top_words = math.ceil(1/2*(len(array)))
#         number_top_words = max_number_top_word
#         list_of_indices_min_value = list(array.argsort()[:number_top_words])
#     else:
#         list_of_indices_min_value = list(array.argsort()[:max_number_top_word])
#     return list_of_indices_min_value

def min_score(array):
    
    # start = time.time()
    
    number_top_words          = max(math.ceil(len(array)),40)
    
    list_of_indices_min_value = list(array.argsort()[:number_top_words])  # 5
    
    # end = time.time()
    # print('Min Score               in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return list_of_indices_min_value

def Corresponding_words(lst_words, list_indices):
    
    # start = time.time()
    
    lst = [lst_words[index] for index in list_indices]
        
    # end = time.time()
    # print('Corresponding Words     in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return lst

def detect_ner(keywords_list, ner_list):
    
    # start = time.time()
    
    new_keywords_list = keywords_list.copy()
    for word in keywords_list:
        ner_is_keyword = [word in entity for entity in ner_list]
        
        if sum(ner_is_keyword):
            new_keywords_list.remove(word)
            new_keywords_list += list(compress(ner_list, ner_is_keyword))
        
    # end = time.time()
    # print('Detect NER              in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return new_keywords_list

def detect_n_gram(keywords_list, n_gram_list):
    
    # start = time.time()
    
    new_keywords_list = keywords_list.copy()
    for word in keywords_list:
        Ngram_is_keyword = [word in Ngram for Ngram in n_gram_list]

        if sum(Ngram_is_keyword):
            new_keywords_list.remove(word)
            new_keywords_list += list(compress(n_gram_list, Ngram_is_keyword))
        
    # end = time.time()
    # print('Detect N Gram           in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return new_keywords_list

def combination_ner_ngram(keywords_list, ner_list, n_gram_list):
    
    # start = time.time()
    
    new_keywords_list = keywords_list.copy()
    for word in keywords_list:
        ner_is_keyword   = [word in entity for entity in ner_list   ]
        Ngram_is_keyword = [word in Ngram  for Ngram  in n_gram_list]
        
        if   sum(ner_is_keyword  ):
            new_keywords_list.remove(word)
            new_keywords_list += list(compress(ner_list   , ner_is_keyword  ))
        elif sum(Ngram_is_keyword):
            new_keywords_list.remove(word)
            new_keywords_list += list(compress(n_gram_list, Ngram_is_keyword))
        
    # end = time.time()
    # print('Combination NER NGram   in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return new_keywords_list

def final_keyword_function(lst):
    
    # start = time.time()
    
    max_number_top_word = 20
    if len(lst) <= 5:
        lst = lst[0:1                  ]
    elif 5< len(lst)<40:
        lst = lst[0:int(0.5*len(lst))  ]
    else:
        lst = lst[0:max_number_top_word]
        
    # end = time.time()
    # print('Final KeyWord Function  in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return lst

def unique_phrase(word_lst):
    
    # start = time.time()
    
    #uniqueWords = []
    #for word in word_lst:
    #    if not word in uniqueWords:
    #        uniqueWords.append(word)
    
    uniqueWords = list(set(word_lst))
        
    # end = time.time()
    # print('Unique Phrase           in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
        
    return uniqueWords

def getting_keys_list(dic):
    
    # start = time.time()
    
    data       = list(dic.items())
    an_array   = np.array(data)
    array_keys = an_array[:, 0]
    
    # end = time.time()
    # print('Getting Keys List       in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    return array_keys

def convert_list_to_string(org_list, seperator=' '):
    
    # start = time.time()
    
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    str = seperator.join(org_list)
        
    # end = time.time()
    # print('Convert List To String  in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
        
    return str
