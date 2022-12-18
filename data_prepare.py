import pandas as pd


def feature_selection_and_preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    A function for preprocessing data for training model
    '''
    features = dataset[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].copy()
    features['Relative'] = dataset['SibSp'] + dataset['Parch']
    features['Age'] = features['Age'].fillna(features['Age'].dropna().median())
    features['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
    max_value = features['Fare'].max()
    min_value = features['Fare'].min()
    features['Fare'] = (features['Fare'] - min_value) / (max_value - min_value)
    features['Embarked'] = features['Embarked'].fillna('S')
    features['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    return features


def feature_select(data: pd.DataFrame) -> pd.DataFrame:
    '''
    A function for preprocessing data for evaluation model
    '''
    features = data[
      ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Relative']
    ].copy()
    features['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
    features['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    return features
