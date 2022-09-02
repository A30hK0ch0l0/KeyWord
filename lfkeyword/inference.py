# -*- coding: utf-8 -*-
# +
#import time
#start = time.time()

from .utils import *

#end  = time.time()
#print('Imports                 in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))

# +
import pandas as pd

def infer(lst):
    
    #t = time.time()
    
    #data = pd.DataFrame(lst, columns=['full_text', 'ner_list'])
    data = lst
    
    #print('Creat DataFrame         in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['full_text'] = data['full_text'].apply(regex)
    
    #print('Regex                   in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['full_text'] = data['full_text'].apply(remove_emojies)
    
    #print('Remove Emojies          in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['full_text'] = data['full_text'].apply(fix_hashtags)
    
    #print('Fix HashTags            in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['n_gram'] = data['full_text'].apply(get_n_gram_words)
    
    #print('Get N Gram Words        in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['items'] = data['full_text'].apply(lambda text: ['']+text.split())
    
    #print('Split Texts To Words    in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    #data['text_normal_without_stopword'] = data['items'].apply(convert_list_to_string)
    #data['len'] = data['items'].apply(lambda x: len(x))
    
    data['dict_word_count'] = data['full_text'].apply(word_count)
    
    #print('Word Count              in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    #data = data[data['dict_word_count'] != {}]
    #data[data['dict_word_count'] == {}]
    
    #print('Remove Empty Dictionary in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['array'] = data['dict_word_count'].apply(convertor_dic_for_value)
    
    #print('Convertor Dic For Value in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['distancce_matrix'] = data['items'].apply(lambda lst: distance_matrix(matrix_embeding_words(lst)))
    
    #print('Distance Matrix Embed   in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['score'] = np.nan
    data['score'] = data.apply(lambda x: score_matrix(x['array'], x['distancce_matrix']), axis=1)
    
    #print('Score Matrix            in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    # applying model on
    
    data['lst_indices'] = data['score'].apply(min_score)
    
    #print('Min Score               in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['lst_words'] = data['dict_word_count'].apply(lambda dic: list(dic.keys()))
    
    #print('Dictionary To List      in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['keywords'] = np.nan
    data['keywords'] = data.apply(lambda x: Corresponding_words(x['lst_words'], x['lst_indices']), axis=1)
    
    #print('Corresponding Words     in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    #data['ner_keywords'] = data.apply(lambda x: detect_ner(x['keywords'], x['ner_list']), axis=1)
    
    #print('Detect NER              in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    #data['n_gram_keywords'] = data.apply(lambda x: detect_n_gram(x['keywords'], x['n_gram']), axis=1)
    
    #print('Detect N Gram           in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['combination_keyword'] = np.nan
    data['combination_keyword'] = data.apply(lambda x:
                                  combination_ner_ngram(x['keywords'], x['ner_list'], x['n_gram']), axis=1)
    
    #print('Combination NER NGram   in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['keywords_without_stopword'] = data['combination_keyword'].apply(lambda lst:
                                            [word for word in lst if word not in list_stopwords])
    
    #print('Remove Stop Words       in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['final_keyword'] = data['keywords_without_stopword'].apply(final_keyword_function)
    
    #print('Final KeyWord Function  in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    data['keywords'] = data['final_keyword'].apply(unique_phrase)
    
    #print('Unique Phrase           in {:11.9} Micro Seconds.'.format((time.time() - t)*(10**6)))
    #t = time.time()
    
    return data.drop(columns = ['n_gram'                   , 'items'           , 'dict_word_count'    ,
                                'array'                    , 'distancce_matrix', 'score'              ,
                                'lst_indices'              , 'lst_words'       , 'combination_keyword',
                                'keywords_without_stopword', 'final_keyword'   ]                      )
