import numpy as np
import scipy as sci
import os

a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

a = np.array(a)

a = np.asmatrix(a)

aRow0 = a[0, :]
aRow1 = a[1, :]
aCol0 = a[:, 0]
aCol2 = a[:, 0]

aFirst2Rows = a[0:1, :]

aRow2Col2 = a[2, 2]

aCol0and2 = a[:, [0, 2]]

aDropCol0 = a[:, -0]

aDropRow0 = a[-1, :]

aRowRepeat = a.repeat(2, axis=0)

b = np.asmatrix([[10],
                 [11],
                 [12]])

b = np.linspace(1, 9, 9)
c = np.arange(11, 20, 1)

d = np.stack((b, c), axis=1)

d = np.asmatrix((np.array([b, c]))).T

rx = np.random.randn(10, 2)

ry = np.random.randn(10, 1)

rz = np.random.randn(10, 1)

I = np.eye(len(ry))

from numpy.linalg import inv

rx = np.asmatrix(rx)

ry = np.asmatrix(ry).T

H = rx*inv(rx.T*rx)*rx.T

ones = np.ones((10, 3))
vals = np.array([4, 5, 6])

valsmat = vals*ones

'''New Class 10/10/2018'''
x = np.random.uniform(1, 100, 1000)
e = np.random.uniform(1, 3, 1000)
y = 30 + 2*x + e

import matplotlib.pyplot as plt

constant = np.ones(len(x))

X = np.stack((constant, x), axis=1)

X = np.matrix(X)

y = np.asmatrix(y).T

from numpy.linalg import inv

H = X*inv(X.T*X)*X.T

plt.scatter(np.asarray(y), np.asarray(H*y))

plt.show()

I = np.eye(len(x))

plt.hist(np.asarray((I-H)*y))

b = inv(X.T*X)*X.T*y

print(b)

ehat = (I-H)*y

s2 = np.asscalar((ehat.T * ehat) / (len(X) - X.shape[1]))

varb = s2*inv(X.T*X)

se = np.sqrt(np.diag(varb))

import statsmodels.api as sm

reg = sm.OLS(y, X)

results = reg.fit()

results.params

results.tvalues


import os
import pandas as pd


df = pd.read_csv('/Users/jugalmarfatia/Downloads/TableF3-1.csv')

print(df)