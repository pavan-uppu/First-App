import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


st.title('Class Predictor')

st.write("""
## You can predict the class for Iris, Wine and Breast cancer problems
""")

dataset_name = st.sidebar.selectbox(
    'To Predict',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest','DT','Bagging','AdaBoost_RF','GradientBoost')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def get_classifier(clf_name):
    clf = None
    if clf_name == 'SVM':
        clf = SVC()
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier()
    elif clf_name == 'DT':
        clf = DecisionTreeClassifier(max_depth=5)
    elif clf_name == 'Bagging':
        clf = BaggingClassifier(DecisionTreeClassifier(random_state=1))
    elif clf_name == 'AdaBoost_RF':
        clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(),random_state=1)
    elif clf_name == 'GradientBoost':
        clf = GradientBoostingClassifier(learning_rate=0.01,random_state=1)
    else:
        clf = clf = RandomForestClassifier(random_state=1)
    return clf

# getting classifier name
clf = get_classifier(classifier_name)

# splitting data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Model fitting
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Performance measures
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
count_misclassified = (y_test != y_pred).sum()
st.write('Misclassified samples: {}'.format(count_misclassified))

# Prediction of classes
if dataset_name == 'Iris':
   sl = st.sidebar.number_input("Enter SepalLength, range is ({} , {})".format(X[:,0].min(), X[:,0].max()))
   sw = st.sidebar.number_input("Enter SepalWidth, range is ({} , {})".format(X[:,1].min(), X[:,1].max()))
   pl = st.sidebar.number_input("Enter PetalLength, range is ({} , {})".format(X[:,2].min(), X[:,2].max()))
   pw = st.sidebar.number_input("Enter PetalWidth, range is ({} , {})".format(X[:,3].min(), X[:,3].max()))
   clf = get_classifier(classifier_name)
   if st.sidebar.button('Predict'):
       clf.fit(X_train, y_train)
       X_test[0, 0] = sl
       X_test[0, 1] = sw
       X_test[0, 2] = pl
       X_test[0, 3] = pw
       output = clf.predict(X_test[0,:].reshape(1,-1))
       if output == 2:
           st.write('class is virginica')
       elif output == 1:
           st.write('class is versicolor')
       else:
           st.write('class is setosa')
else:
    st.sidebar.markdown('Prediction feature is Under construction,'
                        ' you can check for Model Performance')




