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
# Lets use SVM with different kernels to study a 2d version of the iris data
#The linear models LinearSVC() and SVC(kernel='linear') can deliver decision boundaries that have small different. This can be a consequence of the following differences:
# The LinearSVC minimizes the squared loss function. The SVC on the other hand minimizes the regular loss function.
# The LinearSVC makes use of the One-vs-All scheme and on the other hand SVC uses the One-vs-One 

# +
from sklearn.datasets import load_iris
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


# -

iris = load_iris()
X = iris.data[:,2:] # 2d collect just petal length and widht
y = iris.target
x1label = iris.feature_names[2:][0]
x2label = iris.feature_names[2:][1]


# +
C = 1.0  # SVM regularization parameter

model0 = svm.SVC(kernel='linear', C=C)
model0_fit = model0.fit(X, y)
model1= svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)
model1_fit = model1.fit(X, y)

# -



# +
titles = ('Linear kernel','Polynomial (degree 3) kernel')



x_min, x_max = X[:, 0].min() - 1,X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1,X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
plt.subplot(2, 2,1)
title = titles[0]
plt.title(title)
Z = model0_fit.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha = 0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.subplot(2, 2,2)
title = titles[1]
plt.title(title)
Z = model1_fit.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha = 0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
# -

from sklearn.metrics import accuracy_score
model0_fit.predict(X)
y_pred = model0_fit.predict(X)
print(model0_fit.__class__.__name__,accuracy_score(y,y_pred))

# +
# Looking at the plots it seems like we did a great job. Lets see the accuracy of each model

models_fit = [model0_fit,model1_fit]
for i in range(0,len(models_fit)):
    title = titles[i]
    model_fit = models_fit[i]
    y_pred = model_fit.predict(X)
    print(title,model_fit.__class__.__name__,round(accuracy_score(y,y_pred),2))
