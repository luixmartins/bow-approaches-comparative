#from importlib.machinery import SourceFileLoader
#td_bert = SourceFileLoader("_bertVectorizer", "../tf_bert/bertVectorizer/_bertVectorizer.py").load_module()
#td_bert.MyClass()

import pandas as pd 
import numpy as np 
import re 
import nltk 
import warnings 
import string 
import spacy 
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
    def __init__(self, path_dataframe) -> None:
        self.STOPWORDS = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.RSLPStemmer() 
        self.df = pd.read_csv(path_dataframe, usecols=['review', 'positive'])
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def preprocessing(self, list_of_texts: list):
        texts_cleaned = []
        
        for txt in list_of_texts:
            text = txt.lower() # apply lowercase 

            text = re.sub(r'\d+', '', text) # remove numbers

            text = ' '.join([word for word in str(text).split() if word not in self.STOPWORDS]) #remove STOPWORDS 

            text = re.sub(r'#\w+', '', text) # remove hash  

            text = re.sub(r'\s+', ' ', text).strip() # remove extra white space left while removing stuff 

            #apply lemmatizer
            tokens = self.nlp(text)
            text = ' '.join([token.lemma_ for token in tokens if token.is_alpha])

            # remove punctuation
            for special in string.punctuation:
                if special in text:
                    text = text.replace(special, '')

            texts_cleaned.append(text)
        
        return texts_cleaned
    
    def cossine_distance(self, x, y):
        distance = cosine(x, y)
        
        if np.isnan(distance):
            return 1 

        return distance 

    def classifier_models(self):
        models = [
            MLPClassifier(),
            KNeighborsClassifier(metric=self.cossine_distance),
            SVC(),
            GaussianNB(),
        ]
        return models


    def textual_representations(self):
        representations = {
            'TF': CountVectorizer(stop_words=self.STOPWORDS),
            'TF-IDF': TfidfVectorizer(stop_words=self.STOPWORDS),
        }

        return representations 

    def main(self):
        '''
        '''
        X_col = 'text'
        y_col = 'label'
        kfold = 10

        clf_models = self.classifier_models()
        textual_representations = self.textual_representations()
        
        self.df.columns = ['text', 'label']
        y = self.df[y_col].values 
        entries = []

        for name, vectorizer in textual_representations.items():
            if name == 'TD-Bert' or name == 'TD-Distilbert': #Create 
                matrix = vectorizer.fit_transform(self.df[X_col])
                X = matrix.values 
            
            elif name == 'Bert-Base' or name == 'Distilbert':
                model = SentenceTransformer(vectorizer)
                X = model.encode(self.df[X_col].to_list())

            else: 
                texts = self.preprocessing(self.df[X_col].to_list())
                matrix = vectorizer.fit_transform(texts)
                X = pd.DataFrame(matrix.todense(), columns = vectorizer.get_feature_names()).values 
            
            
            for model in clf_models:
                try:
                    model_name = model.__class__.__name__
                    accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=kfold)
                    
                    for fold_index, accuracy in enumerate(accuracies):
                        entries.append((model_name, fold_index, accuracy, name))
            
                except:
                    print("Error in:", name, model_name)

        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_index', 'accuracy', 'name'])
        mean = cv_df.groupby(['model_name', 'name']).accuracy.mean()
        std = cv_df.groupby(['model_name', 'name']).accuracy.std()

        result_cv = pd.concat([mean, std], axis=1, ignore_index=True)
        result_cv.columns = ['Mean Accuracy', 'Standard Deviation']

        return result_cv 

if __name__ == '__main__':
    path = 'datasets/sentiment_analyze_data.csv'

    pipeline = Pipeline(path).main() 

    print(pipeline)


