from dtree import *
from utils import *
import pandas as pd

# Information about attributes
attr_dict = {'outlook': {type_key: AttrType.NOM, choices_key: ['sunny', 'overcast', 'rainy']},
             'temperature': {type_key: AttrType.CONT},
             'humidity': {type_key: AttrType.CONT},
             'windy': {type_key: AttrType.BIN, choices_key: [True, False]},
             'CLASS': {type_key: AttrType.CL, choices_key: [True, False]}}

df = pd.read_csv('data.csv')

# Test example
test = {'outlook': ['overcast'], 'temperature': [60], 'humidity': [62], 'windy': [0]}
test_df = pd.DataFrame(data=test)
test_ex = test_df.iloc[0]


def output_tree(criteria_str, criteria):
    print(criteria_str)
    tree = DecisionTree()
    tree.train(df, attr_dict, criteria)
    print(tree)
    print('Predicted class (1=Yes, 0=No):', tree.predict(test_ex), '\n')


output_tree('Information Gain', Criteria.IG)
output_tree('Gain Ratio', Criteria.GR)
output_tree('Gini Impurity', Criteria.GINI)
