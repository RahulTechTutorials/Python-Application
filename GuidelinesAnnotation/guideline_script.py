import logging
logging.basicConfig(filename='/Users/rahuljain/Desktop/Python/Python_Application/Python-Application/GuidelinesAnnotation/example.log',
                    filemode='w',level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
from pyemd import emd
from gensim.similarities import WmdSimilarity
import gensim
from gensim import models, corpora ##, similaties
from gensim.models import Word2Vec
from string import punctuation
import pandas as pd
import re
from re import search
import requests
from bs4 import BeautifulSoup
import nltk
##nltk.download('stopwords')
##nltk.download('punkt')
##conda install -c anaconda gensim
##conda install -c anaconda nltk
##conda install -c conda-forge pyemd


from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import sys
sys.path.extend('/Users/rahuljain/Desktop/Python/Python_Application/Python-Application/GuidelinesAnnotation/corpus/')
import gensim.downloader as api


def read_guideline():
    #creating a list that contains the name of all the text file in your data #folder
    fileNames = []
    fileNames = [f for f in listdir('/Users/rahuljain/Desktop/Python/Python_Application/Python-Application/GuidelinesAnnotation/') if 
     f.endswith('.txt')]
    logging.debug(docLabels)
    #creating a list data that stores the content of all text files in order of their names in docLabels

    data = []
    ##opening all the file from the list of docs and appending the data together
    for doc in fileNames:
        data.append(open('/Users/rahuljain/Desktop/Python/Python_Application/Python-Application/GuidelinesAnnotation/' + doc).read())
    statement_data = ''.join(data)
    ##print(statement_data)
    return statement_data

def url_to_transcript(url):
    ###Returns web content from the url
    page = requests.get(url).text
    soup = BeautifulSoup(page,"lxml")
    text = [p.text for p in soup.find(class_ = "page-content").find_all('p')]
    logging.debug(text)
    text = ''.join(text)
    return text

def data_pre_processing(pre_tokenized_string):
    num_list = list(range(10))
    numbers = ''.join(str(e) for e in num_list)
    period_comma_adjusted_string = re.sub(r'(?<=[.])(?=[A-Z])', r' ', pre_tokenized_string)
    exclusion_list = numbers + ':'
    period_comma_adjusted_string = re.sub('[%s]' % re.escape(exclusion_list),'',period_comma_adjusted_string)
 ##   for i in exclusion_list:
 ##       period_comma_adjusted_string = period_comma_adjusted_string.replace(i,'')
    return period_comma_adjusted_string
    logging.debug(period_comma_adjusted_string)
    
    
def data_post_processing(post_tokenized_lst):
    new_lst = []
    for sent in post_tokenized_lst:
        new_sent = sent.lower()
        new_sent = re.sub(r'\.', r' ', new_sent)
        new_lst.append(new_sent) 
    return new_lst
        

def sent_tokenizer(statement_data):
    logging.debug(statement_data)
    nltk_tokens = nltk.sent_tokenize(statement_data)
    return nltk_tokens


def word_tokenizer(sentence):
    stopword_set = set(stopwords.words('english'))
    split_sentence = sentence.split()
    words = [w for w in split_sentence if w not in stop_words]
    words = sorted(list(set(words)))
    return words

def train_model():
    os.chdir('/Users/rahuljain/Desktop/Python/Python_Application/Python-Application/GuidelinesAnnotation/corpus/')
    df = pd.read_csv('training_ds.csv')
    tok_corp = [nltk.word_tokenize(sent) for sent in df]
    model = gensim.models.Word2Vec(tok_corp,min_count = 1, size =32)
    model.save('word2vec_test_model')

    
if __name__ == '__main__':
    url = 'https://www.propelnonprofits.org/resources/financial-policy-guidelines-example/'
    ##print(url_to_transcript(url))
    ##print(read_guideline())
    ##print(url_to_transcript(url))
    xweb_corpus = data_post_processing(sent_tokenizer(data_pre_processing(url_to_transcript(url))))
    guideline_corpus= data_post_processing(sent_tokenizer(data_pre_processing(read_guideline())))
    ####training the Word2Vec mode.
    train_model()
    ##Loading the pre-trained model
    #word_vectors = api.load("word2vec_test_model")  # load pre-trained word-vectors from gensim-data


    for IDG, sample in enumerate(guideline_corpus):
        sample_wl = word_tokenizer(sample)
        for IDX, xref in enumerate(xweb_corpus):
            xref_wl = word_tokenizer(xref)
            distance = model.wv.wmdistance(sample_wl, xref_wl)
            ##distance = WmdSimilarity(sample_wl, xref_wl,num_best = 1) 
            print (IDG,IDX,'distance = %.4f' % distance)
            
        
        
        
    

    
