from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

class Classification:
    def __init__(self, model=LogisticRegression(), train_data=None, test_data=None):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
    
    def train_classifier_model(self):
        self.text_clf = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))), ('tfidf', TfidfTransformer()), ('clf', self.model)])
        #Grid Search CV
        # self.gs_clf = GridSearchCV(self.text_clf, parameters, n_jobs=-1)
        # self.gs_clf = self.gs_clf.fit(self.train_data['review'], self.train_data['label'])
        # print(self.gs_clf.best_score_, self.gs_clf.best_params_)
        self.text_clf = self.text_clf.fit(self.train_data['review'], self.train_data['label'])
        predicted = self.text_clf.predict(self.test_data['review'])
        # return np.mean(predicted == self.test_data['label']), confusion_matrix(self.test_data['label'], predicted)
        return roc_auc_score(self.test_data['label'], predicted), confusion_matrix(self.test_data['label'], predicted)

    def predict_text(self, user_input):
        self.user_input = user_input
        return self.text_clf.predict([self.user_input])