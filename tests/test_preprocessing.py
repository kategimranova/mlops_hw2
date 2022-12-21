from data_prepare import feature_selection_and_preprocessing
import pandas as pd
from os.path import dirname, abspath

path = dirname(dirname(abspath(__file__)))


def test_feature_select():
    data = pd.read_csv(path + "/data/train.csv", index_col='PassengerId')
    features = feature_selection_and_preprocessing(data)
    assert features.shape[1] == 6

test_feature_select()
