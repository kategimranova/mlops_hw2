from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing_extensions import Literal
from models.model import KnnModel, LogRegModel

app = FastAPI()


@app.get('/')
def model_info():
    '''Return models information'''
    return {
        "name": {"KNN and Logistic Regression models for Titanic problem"}
    }


class Input(BaseModel):
    '''Input for data validation'''
    Pclass: int = Field(..., ge=1, le=3)
    Sex: Literal['male', 'female']
    Age: int = Field(..., ge=1, le=101)
    Fare: float = Field(...)
    Embarked: Literal['C', 'S', 'Q']
    Relative: int = Field(..., gt=0)


class Output(BaseModel):
    '''Output for data validation'''
    label: int
    prediction: str


class KnnParameters(BaseModel):
    '''Pydantic параметры для выбора гиперпараметра для KNN'''
    n_neighbors: int = Field(3, ge=2, le=20)


class LogRegParameters(BaseModel):
    '''Pydantic параметры для выбора гиперпараметра для Логистической Регрессии'''
    penalty: Literal['l1', 'l2', 'none']


@app.on_event("startup")
def load_model():
    '''При запуске сервера загружаются модели KNN и Logistic Regression'''
    global knn_model, logreg_model
    knn_model = KnnModel()
    logreg_model = LogRegModel()


@app.get("/available_models")
def get_available_models():
    '''Список доступных для обучения классов моделей'''
    return {"models": [knn_model.description, logreg_model.description]}


@app.post('/train/knn')
def train_knn(params: KnnParameters):
    '''Обучение модели KNN, можно задать гиперпараметр n_neighbors'''
    global knn_model
    n_neighbors = params.__dict__['n_neighbors']
    knn_model = KnnModel(n_neighbors=n_neighbors)
    knn_model.train()


@app.post('/train/logreg')
def train_logreg(params: LogRegParameters):
    '''Обучение модели Logisctic Regression,
    можно задать гиперпараметр penalty
    '''
    global logreg_model
    penalty = params.__dict__['penalty']
    logreg_model = LogRegModel(penalty=penalty)
    logreg_model.train()


@app.post('/predict/knn', response_model=Output)
def model_predict_knn(input: Input):
    label = knn_model.predict_(input)[0]

    if label == 1:
        prediction = 'Survived'
    else:
        prediction = 'Not survived'

    return {
        'label': label,
        'prediction': prediction
    }


@app.post('/predict/logreg', response_model=Output)
def model_predict_logreg(input: Input):
    '''Получение предсказания по модели Logistic Regression'''
    label = logreg_model.predict_(input)[0]

    if label == 1:
        prediction = 'Survived'
    else:
        prediction = 'Not survived'

    return {
        'label': label,
        'prediction': prediction
    }
