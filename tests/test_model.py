from models.model import KnnModel, LogRegModel
from api import Input
import os


def test_knn_1():
    '''Testing whether model file exists after training'''
    knn_model = KnnModel().train()
    path_to_model = knn_model.path_to_model
    assert os.path.exists(path_to_model)


def test_logreg_1():
    '''Testing whether model file exists after training'''
    logreg_model = LogRegModel().train()
    path_to_model = logreg_model.path_to_model
    assert os.path.exists(path_to_model)


def test_knn_2():
    '''Testing whether prediction returns valid result'''
    knn_model = KnnModel().train()
    input_dict = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 34,
        "Fare": 0,
        "Embarked": "C",
        "Relative": 3
    }
    input = Input(**input_dict)
    result = knn_model.predict_(input)
    assert result in [0, 1]


def test_logreg_2():
    '''Testing whether prediction returns valid result'''
    logreg_model = LogRegModel().train()
    input_dict = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 34,
        "Fare": 0,
        "Embarked": "C",
        "Relative": 3
    }
    input = Input(**input_dict)
    result = logreg_model.predict_(input)
    assert result in [0, 1]
