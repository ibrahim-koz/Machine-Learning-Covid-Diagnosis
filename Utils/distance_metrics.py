import pandas as pd
import scipy.linalg
import numpy as np


def euclidean_distance(instance1, instance2, attributes):
    sum_dist = 0
    for i in attributes:
        sum_dist += (int(instance1[i]) - int(instance2[i])) ** 2
    return sum_dist


def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(np.transpose(data))
    inv_covmat = scipy.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv'
df = pd.read_csv(filepath).iloc[:, [0, 4, 6]]
df.head()

df_x = df[['carat', 'depth', 'price']].head(500)
ax = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
df_x.head()
