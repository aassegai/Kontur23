import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm 
from pymorphy2 import MorphAnalyzer

morph_analyzer = MorphAnalyzer()
stopwords_rus = stopwords.words('russian')
punctuation_rus = ['«', '»', "''", '``', '„', '“', '..', '...', '—', '–', '…', '’', '№']
stopwords_rus = set(stopwords_rus) | set(punctuation_rus)

class DataPreprocessor:
    def __init__(self, lemmatizer=morph_analyzer,
                 remove_punctuation=False,
                 lemmatize=False,
                 for_rnn=False,
                 remove_stopwords=False):
        self.lemmatizer = lemmatizer
        self.remove_punctuation = remove_punctuation
        self.lemmatize = lemmatize
        self.for_rnn = for_rnn
        self.remove_stopwords = remove_stopwords

    def preprocess_text(self, texts, annotations):
        new_texts = []
        new_annotations = []
        for i in tqdm(texts.index):
            temp_text = texts[i].strip().replace('\n', ' ').lower()
            if self.remove_punctuation:
                temp_text = re.sub(r'[^\w\s]', ' ', temp_text)
            prep_text = ''
            for word in temp_text.split():
                if self.remove_stopwords and word not in stopwords_rus:  
                        if self.lemmatize:
                            prep_text = prep_text + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                        else:
                            prep_text = prep_text + '' + word + ' '
                else:  
                    if self.lemmatize:
                        prep_text = prep_text + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                    else:
                        prep_text = prep_text + '' + word + ' '
            # if self.for_rnn:
            #     prep_text = '<BOS>' + prep_text + '<EOS>'
            new_texts.append(prep_text)


            # print(annotations[i]['result'])      
            segments = []
            new_segment = {}
            temp_segment = annotations[i]['text'][0].strip().replace('\n', ' ').lower()
            if self.remove_punctuation:
                temp_segment = re.sub(r'[^\w\s]', ' ', temp_segment)
            prep_segment = ''
            for word in temp_segment.split():
                if self.remove_stopwords and word not in stopwords_rus:  
                        if self.lemmatize:
                            prep_text = prep_text + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                        else:
                            prep_text = prep_text + '' + word + ' '
                else:  
                    if self.lemmatize:
                        prep_text = prep_text + '' + self.lemmatizer.parse(word)[0].normal_form + ' '
                    else:
                        prep_text = prep_text + '' + word + ' '            
            prep_segment = prep_segment.strip()
            if self.for_rnn:
                prep_segment = '<BOS>' + prep_segment + '<EOS>'
            # print(prep_segment)
            new_segment['answer_start'] = [prep_text.find(prep_segment)]
            new_segment['answer_end'] = [prep_text.find(prep_segment) + len(prep_segment)]
            new_segment['text'] = [prep_segment]
            segments.append(new_segment)
            new_annotations.append(segments)
            
        
        return new_texts, new_annotations



    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        self.dataframe = df
        new_texts, new_annotations = self.preprocess_text(df.text, df.extracted_part)
        
        self.dataframe.drop_duplicates(subset='id', inplace=True)
        self.dataframe = self.dataframe.set_index('id')

        for i, idx in enumerate(self.dataframe.index):
            self.dataframe.loc[idx].text = new_texts[i]
            self.dataframe.loc[idx].extracted_part = new_annotations[i][0]
        return self.dataframe
        