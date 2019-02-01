from string import punctuation
import re
from re import search
import requests
from bs4 import BeautifulSoup
import nltk
##nltk.download('stopwords')
##nltk.download('punkt')
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join


def read_guideline():
    #creating a list that contains the name of all the text file in your data #folder
    fileNames = []
    fileNames = [f for f in listdir('/Users/rahuljain/Desktop/Python/Python_Application/GuidelinesAnnotation/') if 
     f.endswith('.txt')]
    ##printing the list of all the files
    ##print(docLabels)
    #creating a list data that stores the content of all text files in order of their names in docLabels

    data = []
    ##opening all the file from the list of docs and appending the data together
    for doc in fileNames:
        data.append(open('/Users/rahuljain/Desktop/Python/Python_Application/GuidelinesAnnotation/' + doc).read())
    statement_data = ''.join(data)
    return statement_data

def tokenizer(statement_data):
    ##print(statement_data)
    nltk_tokens = nltk.sent_tokenize(statement_data)
    ##print(type(nltk_tokens))
    return nltk_tokens

def data_pre_processing(pre_tokenized_string):
    num_list = list(range(10))
    numbers = ''.join(str(e) for e in num_list)
    period_comma_adjusted_string = re.sub(r'(?<=[.])(?=[A-Z])', r' ', pre_tokenized_string)
    exclusion_list = numbers + ':'
    period_comma_adjusted_string = re.sub('[%s]' % re.escape(exclusion_list),'',period_comma_adjusted_string)
 ##   for i in exclusion_list:
 ##       period_comma_adjusted_string = period_comma_adjusted_string.replace(i,'')
    return period_comma_adjusted_string
    ##print(period_comma_adjusted_string)
    
    
def data_post_processing(post_tokenized_lst):
    ##stopword_set = set(stopwords.words('english'))
    new_lst = []
    for sent in post_tokenized_lst:
        new_sent = sent.lower()
        new_lst.append(new_sent)
        
    return new_lst
        


def url_to_transcript(url):
    ###Returns web content from the url
    page = requests.get(url).text
    soup = BeautifulSoup(page,"lxml")
    text = [p.text for p in soup.find(class_ = "page-content").find_all('p')]
    ##print(url)
    ##print(text)
    text = ''.join(text)
    return text

    
if __name__ == '__main__':
    url = 'https://www.propelnonprofits.org/resources/financial-policy-guidelines-example/'
    ##print(url_to_transcript(url))
    ##tokenizer(read_guideline())
    ##tokenizer(url_to_transcript(url))
    print(data_post_processing(tokenizer(data_pre_processing(url_to_transcript(url)))))
    ##data_post_processing(tokenizer(data_pre_processing(url_to_transcript(url))))
    ##stopword_set = set(stopwords.words('english'))
    ##print(type(stopword_set))
    ##print(stopword_set)

    
