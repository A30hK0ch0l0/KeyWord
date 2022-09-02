# -*- coding: utf-8 -*-
# +
import time
start  = time.time()

import pickle
import pandas as pd
import os
import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from lfkeyword import infer

end    = time.time()
print('Imports            in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))


# -

# applying keyword extraction on a sample data
def test_keyword():
    lst = [{'full_text': 'اشالله با وتصویب انهم انقراض این حیوانات دوپااغازمی گردد.',
            'ner_list' : ['تاجگردون'                                              ]},
           {'full_text': 'معنی ایه سوره توبه خدا اهل ایمان را از مرد و زن وعده فرموده که در بهشت خلد ابدی که زیر درختانش نهرها جاری است دراورد و در عمارات نیکو و پاکیزه بهشت عدن منزل دهد و برتر و بزرگتر از هر نعمت مقام رضا و خشنودی خداست و ان به حقیقت فیروزی بزرگ است.',
            'ner_list' : ['بهشت عدن', 'مقام رضا', 'فیروزی', 'جنت عدن', 'سوره'     ]},
           {'full_text': 'این مقام ایرانی تأکید کرده است اگر در مذاکرات توافقی حاصل شود،‌ ایران مطمئناً تصاویر دوربین‌های نظارتی را به آژانس انرژی هسته‌ای تحویل می‌دهد. بنا به نوشته این رسانه آمریکایی، ایران با این تصمیم تیم‌های مذاکره‌کننده در وین برای احیای برجام را تحت فشار قرار می‌دهد.',
            'ner_list' : ['آژانس انرژی هسته‌ای'                                   ]},
           {'full_text': 'مالیات بر مسکن‌های خالی رو افزایش دهید تا هم قیمت مسکن پایین بیاید و هم اجاره‌ها- رونوشت به مسئولین نیمه محترم دولت و مجلس https://t.co/qdjhlr22ph.',
            'ner_list' : ['مسئولین نیمه محترم دولت و مجلس'                        ]},
           {'full_text': 'با گذشت 13 سال از پیروزی انقلاب یعنی در سال 1370 مجددا کاخ مرمر را برای ایجاد تغییراتی کاندید کرده و این عمارت را ابتدا به قوه قضائیه و نهایتا به مجمع تشخیص مصلحت نظام واگذار شده و از آن زمان تا کنون در نامه های دولتی از این کاخ با عنوان ساختمان قدس یاد می شود. پس از آن نیز در سال 1397 توسط این مجمع به بنیاد مستضعفین تحویل داده شد و یک سال بعد پس از آن که عملیات مورد نیاز برای تغییر کاربری روی آن انجام شد، در سال 98 به عنوان موزه در اختیار عموم مردم قرار گرفت. نکته بسیار جالب درباره کاخ مرمر تهران این است که این بنا، بعد 41 سال در بهمن ماه سال 98 برای اولین بار مورد بازدید رسانه ها قرار گرفت.',
            'ner_list' : ['کاخ مرمر تهران', 'مجمع تشخیص مصلحت نظام'               ]}]
    
    full_texte = ' '
    ner_liste  = [' ']
    
    lst.append({'full_text': full_texte,
                'ner_list' : ner_liste })
    
    
    full_text = ''
    ner_list  = []
    for dict in lst:
        full_text += dict['full_text']
        ner_list  += dict['ner_list' ]
    
    lst.append({'full_text': full_text,
                'ner_list' : ner_list })
    
    df     = pd.DataFrame(lst)
    
    start  = time.time()
    
    output = infer(df)
    
    print(output)
    
    end    = time.time()
    print('KeyWords Extracted      in {:11.9} Micro Seconds.'.format((end - start)*(10**6)))
    
    data_path = f'{os.path.dirname(os.path.abspath(__file__))}'
    
    #output.to_pickle(f'{data_path}/test.pickle')
    
    test = pd.read_pickle(f'{data_path}/test.pickle')
    
    assert((output['keywords'].apply(lambda l: sorted(l)) ==
              test['keywords'].apply(lambda l: sorted(l))).all())
