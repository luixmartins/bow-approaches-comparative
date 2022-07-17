#from importlib.machinery import SourceFileLoader
#td_bert = SourceFileLoader("_bertVectorizer", "../tf_bert/bertVectorizer/_bertVectorizer.py").load_module()
#td_bert.MyClass()

import pandas as pd 
import numpy as np 
import re 
import nltk 
import warnings 
#TEXTUAL REPRESENTATIONS 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertVectorizer import bertVectorizer as td_bert 
from sentence_transformers import SentenceTransformer
#CLASSIFIERS 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#EVALUATE METRICS 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')
nltk.download('rslp')
nltk.download('stopwords')
nltk.download('punkt')

class Pipeline:
    def __init__(self) -> None:
        self.STOPWORDS = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.RSLPStemmer 

    
    def main(self, df):
        pass 
