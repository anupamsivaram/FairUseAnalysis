get_ipython().system('pip install joblib')
get_ipython().system('pip install xgboost')
import joblib
import json
print("Done.")


# In[3]:


from IPython.utils import io

import numpy as np
import pywt
import cv2
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams["figure.figsize"] = (20,10)
import pandas as pd
import seaborn as sn

import os
import shutil

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

print("Done.")


df1 = pd.read_csv('/Users/anupamsivaram/Desktop/Amended Fair Use Findings.csv')
df1.head


# # Data Cleaning


df1.shape
df1.groupby('outcome')['outcome'].agg('count')


# In[24]:


import pandas as pd
import re

def categorize_outcome(text):
    text_lower = text.lower()
    
    if re.search(r'\bfair use found\b', text_lower):
        return '1'
    elif re.search(r'\bfair use not found\b', text_lower):
        return '0'
    else:
        return 'UNCATEGORIZED'

df1['outcome2'] = df1['outcome'].apply(categorize_outcome)

df1.groupby('outcome2')['outcome2'].agg('count')


# In[25]:


df2 = df1.drop(['case_number','key_facts','issue','tags'],axis='columns')
df2.head


# In[26]:


# As part of data cleaning, we'll ensure that no values have been omitted.

df2.isnull().sum()

df1.groupby('court')['court'].agg('count')


import pandas as pd
import re

def categorize_circuit(text):
    text_lower = text.lower()
    
    if re.search(r'\bsupreme court\b', text_lower):
        return 'Supreme Court'
    elif re.search(r'\bdistrict court\b', text_lower):
        return 'District Court'
    elif re.search(r'\bfirst circuit\b', text_lower):
        return 'First Circuit'
    elif re.search(r'\bsecond circuit\b', text_lower):
        return 'Second Circuit'
    elif re.search(r'\bthird circuit\b', text_lower):
        return 'Third Circuit'
    elif re.search(r'\bfourth circuit\b', text_lower):
        return 'Fourth Circuit'
    elif re.search(r'\bfifth circuit\b', text_lower):
        return 'Fifth Circuit'
    elif re.search(r'\bsixth circuit\b', text_lower):
        return 'Sixth Circuit'
    elif re.search(r'\bseventh circuit\b', text_lower):
        return 'Seventh Circuit'
    elif re.search(r'\beighth circuit\b', text_lower):
        return 'Eighth Circuit'
    elif re.search(r'\bninth circuit\b', text_lower):
        return 'Ninth Circuit'
    elif re.search(r'\btenth circuit\b', text_lower):
        return 'Tenth Circuit'
    elif re.search(r'\beleventh circuit\b', text_lower):
        return 'Eleventh Circuit'
    else:
        return 'UNCATEGORIZED'

df2['circuit'] = df2['court'].apply(categorize_circuit)

df2.groupby('circuit')['circuit'].agg('count')


df2[df2.circuit=='UNCATEGORIZED']

def categorize_market(text):
    text_lower = text.lower()
    
    if re.search(r'\bmarket\b', text_lower):
        return 1
    else:
        return 0
    
def categorize_purpose(text):
    text_lower = text.lower()
    
    if re.search(r'\bpurpose\b', text_lower):
        return 1
    else:
        return 0
    
def categorize_transformative(text):
    text_lower = text.lower()
    
    if re.search(r'\btransformative\b', text_lower):
        return 1
    else:
        return 0
    
def categorize_commercial(text):
    text_lower = text.lower()
    
    if re.search(r'\bcommercial\b', text_lower):
        return 1
    else:
        return 0
    
def categorize_harm(text):
    text_lower = text.lower()
    
    if re.search(r'\bharm\b', text_lower):
        return 1
    else:
        return 0
    
df2['market'] = df2['holding'].apply(categorize_market)
df2['purpose'] = df2['holding'].apply(categorize_purpose)
df2['transformative'] = df2['holding'].apply(categorize_transformative)
df2['commercial'] = df2['holding'].apply(categorize_commercial)
df2['harm'] = df2['holding'].apply(categorize_harm)

print(df2.groupby('market')['market'].agg('count'))
print(df2.groupby('purpose')['purpose'].agg('count'))
print(df2.groupby('transformative')['transformative'].agg('count'))
print(df2.groupby('commercial')['commercial'].agg('count'))
print(df2.groupby('harm')['harm'].agg('count'))


# # Model Preparation

X = df2[['market', 'purpose', 'transformative', 'commercial', 'harm']]
y = df2['outcome2']

y_train = y_train.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# # Model Building and Tuning


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd

y_train = y_train.astype(int)

model_params = {
    'linear_regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'lasso': {
        'model': Lasso(),
        'params': {}
    },
    'svm': {
        'model': SVC(probability=True),
        'params': {
            'svc__C': [1, 10, 100, 1000],
            'svc__kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'logisticregression__C': [1, 5, 10]
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    'decision_tree': {
        'model': tree.DecisionTreeClassifier(random_state=1),
        'params': {
            'decisiontreeclassifier__max_depth': [None, 5, 10]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'kneighborsclassifier__n_neighbors': [3, 5, 7],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    },
    'xgboost': {
        'model': XGBClassifier(),
        'params': {
            'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
            'xgbclassifier__n_estimators': [50, 100, 200]
        }
    }
}

scores = []
best_estimators = {}

for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

scoreResults = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
scoreResults


# In[57]:


best_estimators['svm'].score(X_test.astype(int), y_test.astype(int))


# In[58]:


best_estimators['random_forest'].score(X_test.astype(int), y_test.astype(int))


# In[59]:


best_estimators['logistic_regression'].score(X_test.astype(int),y_test.astype(int))


# In[60]:


best_estimators['xgboost'].score(X_test.astype(int),y_test.astype(int))


# In[62]:


best_clf = best_estimators['xgboost']

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype(int), best_clf.predict(X_test.astype(int)))
cm


# In[63]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
