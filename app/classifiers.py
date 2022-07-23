import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine

class Clasifiers:
    def __init__(self, X_values, y_values, representation, kfolds=10) -> None:
        self.X = X_values 
        self.y = y_values 
        self.representation = representation 
        self.kfold = kfolds 


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
    
    def make_prediction(self):
        classifiers = self.classifier_models()
        entries = []
        
        for model in classifiers:
            name = model.__class__.__name__
            accuracies = cross_val_score(model, self.X, self.y, scoring='accuracy', cv=self.kfold)

            for fold_index, accuracy in enumerate(accuracies):
                entries.append((name, fold_index, accuracy))
        
        df_validation = pd.DataFrame(entries, columns=['model_name', 'fold_index', 'accuracy'])

        mean = df_validation.groupby(['model_name', 'name']).accuracy.mean()
        std = df_validation.groupby(['model_name', 'name']).accuracy.std()

        df = pd.concat([mean, std], axis=1, ignore_index=True)
        df.columns = ['Mean Accuracy', 'Standard Deviation']

        df.to_csv(f'./datasets/results/{self.representation}.csv')
