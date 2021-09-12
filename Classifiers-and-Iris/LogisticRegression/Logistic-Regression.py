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

# +
# Lets use a logistic regreesion

# +
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np


# -

iris = load_iris()
X = iris.data[:,2:] # 2d collect just petal length and widht
y = iris.target
x1label = iris.feature_names[2:][0]
x2label = iris.feature_names[2:][1]


log_clf = LogisticRegression()
log_clf_fit = log_clf.fit(X, y)


# +

x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
plt.title("Logistic Regression")
Z = log_clf_fit.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha = 0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')


# +
# looking at the plot we can expect a good accuracy of this model. The boundaries are linear but well placed.
from sklearn.metrics import accuracy_score


y_pred = log_clf_fit.predict(X)
print(log_clf_fit.__class__.__name__,accuracy_score(y,y_pred))

# accuracy of 0.97!
