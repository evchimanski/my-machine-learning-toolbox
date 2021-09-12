# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Lets use a voting system to increase the accuracy of classification problems.
# We can use the classifiers we created (see other projects:
# SVM, Logistic Regression, DecisionTree and Random Forest


# +
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import numpy as np

# -

iris = load_iris()
X = iris.data[:,2:] # 2d collect just petal length and widht
y = iris.target
x1label = iris.feature_names[2:][0]
x2label = iris.feature_names[2:][1]


# +
models_fit =[]
titles = []

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf_fit = tree_clf.fit(X,y)
models_fit.append(tree_clf_fit)
titles.append("DecisionTree")
rnd_clf = RandomForestClassifier(n_estimators = 500,max_leaf_nodes =7,n_jobs = 2)
rnd_clf_fit = rnd_clf.fit(X,y)
models_fit.append(rnd_clf_fit)
titles.append("RandomForest")

C = 1.0  # SVM regularization parameter
svm_linear_clf = svm.SVC(kernel='linear', C=C)
svm_linear_clf_fit = svm_linear_clf.fit(X, y)
models_fit.append(svm_linear_clf_fit)
titles.append("SVMLinear")

svm_poly3= svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)
svm_poly3_fit = svm_poly3.fit(X, y)
models_fit.append(svm_poly3_fit)
titles.append("SVMPoly3")

log_clf = LogisticRegression()
log_clf_fit = log_clf.fit(X, y)
models_fit.append(log_clf_fit)
titles.append("LogisticRegression")

voting_clf = VotingClassifier(
    estimators = [(titles[0],tree_clf),(titles[1],rnd_clf),(titles[2],svm_linear_clf),(titles[3],svm_poly3),(titles[4],log_clf)],
    voting ='hard')


voting_clf_fit = voting_clf.fit(X, y)
models_fit.append(voting_clf_fit)

titles.append("VotingClassifier")


# -

titles

# Lets see the accuracy
for i in range(0,len(models_fit)):
    title = titles[i]
    model_fit = models_fit[i]
    y_pred = model_fit.predict(X)
    print(title,model_fit.__class__.__name__,round(accuracy_score(y,y_pred),2))

# +
# Here we have seen that the voting classifier has an accuracy slightly better then the averaged accuracy among the individual classifiers
# Note that we have the Random Forest taht is probably overfitting the data since we have very deep trees in there
# In this way we can expect the voting system to smooth that down and split the resposability of classification among the different models.

# Most of the times when the individual classifiers perform as 0.8 - 0.9 the voting system can outperform all of them and increase the accuracy of our
# redictions
