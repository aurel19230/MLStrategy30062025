# import pandas for data wrangling
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

# import numpy for Scientific computations
import numpy as np

# import machine learning libraries
import xgboost as xgb
from sklearn.metrics import accuracy_score

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# import packages for feature importance
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt

# import seaborn for correlation heatmap
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#data = '/kaggle/input/wholesale-customers-data-set/Wholesale customers data.csv'
file_path = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/example/Wholesale customers data.csv"

df = pd.read_csv(file_path)

X = df.drop('Channel', axis=1)

y = df['Channel']
X.head()
y.head()
# convert labels into binary values

y[y == 2] = 0

y[y == 1] = 1

