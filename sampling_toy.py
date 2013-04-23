import numpy as np
import math
import random
from matplotlib.pylab import *
from matplotlib.patches import Ellipse

K = 2

def prepare_data(K):
    centers = np.array([[0,0],
                        [10,10]])
    cov = np.array([[[5,0],
                     [0,3]],
                    [[3,0],
                     [0,9]]])
    X = []
    for i in range(K):
        Xi = np.random.multivariate_normal(centers[i], cov[i], 50 )
        X.append(Xi)
    return np.vstack(X), centers, cov


def sample(prob):
    r = np.random.rand()
    p = 0
    for i in range(len(prob)):
        p += prob[i]
        if r < p:
            return i

def predictive(xi, X):
#    print '-----------'
#    print 'X shape = ' , X.shape
    X = np.mat(X)
    mu = X.mean(axis=0)
    cov = 1./X.shape[0] * (X - np.tile(mu, (X.shape[0], 1))).T * (X - np.tile(mu, (X.shape[0], 1)))
#    print 'mu: ', mu
#    print 'cov: ', cov
#    print np.linalg.det(2 * np.pi * cov)
    norm_const = np.linalg.det(2 * np.pi * cov) ** (-0.5)
    return norm_const * np.exp(-0.5 * (xi-mu) * cov.I * (xi-mu).T)
    
def gibbs_sampling(X):
    N = X.shape[0]
    k = 2
    z = np.random.randint(k, size=(N, 1))
# prior of Gaussian
#    mu0 = X.mean(axis=0)
    for it in range(100):
        print it
        for i in range(X.shape[0]):
            prob = np.zeros(k) 
            for t in range(k):
                num = (z==t).sum() - (z[i]==t) 
                prob[t] = float(num) / (N-1)  # from CRP
                prob[t] *= predictive(X[i], X[np.nonzero(z == t)[0], :])
            prob /= prob.sum()
            z[i] = sample(prob)
        if it % 10 == 0:
            visualize_data(X, None, None, z=z)
    return z


def visualize_data(X, centers, cov, z=None):
    clf()
    if z is None:
        plot(X[:,0], X[:,1], '.')
        for i in range(len(centers)):
            error_ellipse(centers[i], cov[i])
        show()
    else:
        plot(X[np.nonzero(z==0)[0], 0], X[np.nonzero(z==0)[0], 1], '.')
        plot(X[np.nonzero(z==1)[0], 0], X[np.nonzero(z==1)[0], 1], 'x')
        show()


def error_ellipse(mu, cov, ax=None):
    """
    Plot the error ellipse at a point given it's covariance matrix.
    """
    # some sane defaults
    facecolor = 'none'
    edgecolor = 'k'
    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
            width=2 * np.sqrt(S[0]),
            height=2 * np.sqrt(S[1]),
            angle=theta,
            facecolor=facecolor, edgecolor=edgecolor)

    if ax is None:
        ax = matplotlib.pyplot.gca()
    ax.add_patch(ellipsePlot)


def main():
    X, centers, cov = prepare_data(K)
    z = gibbs_sampling(X)
    visualize_data(X, centers, cov, z=z)
    

if __name__ == '__main__':
    main()
