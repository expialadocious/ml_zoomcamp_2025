import pandas as pd
import numpy as np
import re
import datetime as dt
from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

### create Sklearn compatible XGBClassifier
from xgboost import XGBClassifier

## import pipeline
from sklearn.pipeline import make_pipeline

Import pickle

## Read ckd_data.csv file (derived from UCI data fetch)
df = pd.read_csv('ckd_data.csv')
try:
    del df['Unnamed: 0']
except:
    pass

## Strip WhiteSpace
df = df.map(lambda x: x.strip() if isinstance(x, str) else x )

## save median values for "false" ckd to a dictionary
    ## will be used to map onto Null values where applicable
ckd_median = (df[df['class']=="ckd"]).describe().iloc[5,:]
ckd_median = ckd_median.to_dict()

## save median values for "false" ckd to a dictionary
    ## will be used to map onto Null values where applicable
notckd_median = (df[df['class']=="notckd"]).describe().iloc[5,:]
notckd_median = notckd_median.to_dict()

true_pct = df['class'].value_counts().iloc[0] / df.shape[0]
false_pct = df['class'].value_counts().iloc[1] / df.shape[0]

## CALCULATE NEW DICTIONARY WITH WEIGHTED MEDIAN VALUES WHEN "TRUE" AND "FALSE" PRESENT FOR CKD
weighted_fill_values = {}

for key in ckd_median.keys():
    weighted_median = (ckd_median[key] * true_pct) +\
    (notckd_median[key] * false_pct)

    weighted_fill_values[key] = weighted_median

## Fill Null values with weighted median calculated values for numerical Null occurrences per column
df = df.fillna(value=weighted_fill_values)

## Remove Null Values in Categorical Columns
    ##See Notebook.ipynb for research and decision to remove
del df['rbc']
del df['pc']
### Delete remaining rows with Null Values in Categorical Columns
df = df.dropna()

## save progress
df1 = df.copy()

### Prepare predictor variable
df1['class'] = np.where((df1['class']=="ckd"), '1', '0')

df1['class']= df1['class'].astype(int)

### Train/Test Splits  (Train, Validation, Test set)
df_full_train, df_test = train_test_split(df1, test_size=0.2, random_state=3)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=3)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['class'].values
y_val = df_val['class'].values
y_test = df_test['class'].values

del df_train['class']
del df_val['class']
del df_test['class']

### BUILD XGBOOST WINNING MODEL
## create dictionaries for all feature datasets
train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

### Create Model with winning parameters
xgb_clf = XGBClassifier(
    learning_rate=0.1,        # eta
    max_depth=10,
    min_child_weight=1,
    n_estimators=150,          # num_boost_round
    objective='binary:logistic',
    n_jobs=8,                  # nthread
    random_state=1,            # seed
    verbosity=1,
)

## Create Pipeline
pipeline = make_pipeline(
    DictVectorizer(sparse=False),
    xgb_clf
)

### Train XGBClassifer model
pipeline.fit(train_dicts, y_train)

## Save out model using Pickle
with open("pipeline_v2.bin", "wb") as f_out:
    pickle.dump(pipeline, f_out)