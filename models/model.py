from data_prepare import feature_selection_and_preprocessing, feature_select
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
from os.path import dirname, abspath
from sklearn.model_selection import train_test_split

path = dirname(dirname(abspath(__file__)))


class KnnModel:
    '''
    A class to repesent a KNN model for Titanic problem.
    ...
    Attributes
    ----------
    Methods:
    train: training model
    predict_: predicting result
    _set_model_path: setting path to a model
    get_parameters: getting parameters of a KNN model (n_neighbors)
    '''
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.path_to_model = path + '/models/knn/knn_model.pkl'
        self.data = pd.read_csv(path + "/data/train.csv", index_col='PassengerId')
        self.data_train, self.data_test = train_test_split(
            self.data, test_size=200, random_state=42
        )
        self.description = "KNN model"

    def train(self):
        if os.path.exists(self.path_to_model):
            os.remove(self.path_to_model)
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors
        )
        self.model.fit(
            feature_selection_and_preprocessing(
                self.data_train.drop('Survived', axis=1)
            ),
            self.data_train['Survived']
        )
        with open(self.path_to_model, 'wb') as f:
            pickle.dump(self.model, f)
        return self

    def predict_(self, input):
        if os.path.exists(self.path_to_model):
            self.model = pickle.load(open(self.path_to_model, "rb"))
        else:
            self.train()
        data_ = pd.DataFrame(input.__dict__, index=[0])
        res = self.model.predict(feature_select(data_))
        return res

    def _set_model_path(self, path_to_model):
        self.path_to_model = path_to_model

    def get_parameters(self):
        return {"n_neighbors": self.n_neighbors}

    def __str__(self):
        return f"KNN model, number of nearest neighbors = {self.n_neighbors}"


class LogRegModel:
    '''
    A class to repesent a Logistic Regression model for Titanic problem.
    ...
    Attributes
    ----------
    Methods:
    train: training model
    predict_: predicting result
    _set_model_path: setting path to a model
    get_parameters: getting parameters of a Logistic regression model (penalty)
    '''
    def __init__(self, penalty='l2'):
        self.penalty = penalty
        self.path_to_model = path + '/models/logreg/logreg_model.pkl'
        self.data = pd.read_csv(path + "/data/train.csv", index_col='PassengerId')
        self.data_train, self.data_test = train_test_split(
            self.data, test_size=200, random_state=42
        )
        self.description = "Logistic Regression model"

    def train(self):
        if os.path.exists(self.path_to_model):
            os.remove(self.path_to_model)
        self.model = LogisticRegression(
            penalty=self.penalty, solver='saga'
        )
        self.model.fit(
            feature_selection_and_preprocessing(
                self.data_train.drop('Survived', axis=1)
            ),
            self.data_train['Survived']
        )
        with open(self.path_to_model, 'wb') as f:
            pickle.dump(self.model, f)
        return self

    def predict_(self, input):
        if os.path.exists(self.path_to_model):
            self.model = pickle.load(open(self.path_to_model, "rb"))
        else:
            self.train()
        data_ = pd.DataFrame(input.__dict__, index=[0])
        res = self.model.predict(feature_select(data_))
        return res

    def _set_model_path(self, path_to_model):
        self.path_to_model = path_to_model

    def get_parameters(self):
        return {"penalty": self.penalty}

    def __str__(self):
        return f"Logistic Regression model, penalty = {self.penalty}"
