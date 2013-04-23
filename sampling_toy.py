import numpy as np
import math
import random
from matplotlib.pylab import *
from matplotlib.patches import Ellipse
from scipy.special import gamma

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
    X = np.mat(X)
    n = X.shape[0]
    d = X.shape[1]
    '''
    mu = X.mean(axis=0)
    cov = 1./n * (X - np.tile(mu, (n, 1))).T * (X - np.tile(mu, (n, 1)))
    norm_const = np.linalg.det(2 * np.pi * cov) ** (-0.5)
    return norm_const * np.exp(-0.5 * (xi-mu) * cov.I * (xi-mu).T)
    '''
    mu = X.mean(axis=0)
    cov = 1./n * (X - np.tile(mu, (n, 1))).T * (X - np.tile(mu, (n, 1)))

    mu0 = np.array([0,0])
    k0 = 1
    v0 = 2
    cov0 = np.eye(2) * 5

    kn = k0 + 1
    vn = v0 + n
    mun = (k0*mu0 + n*mu) / kn 
    covn = cov0 + cov + k0*n / (k0 + n) * (mu - mu0) * (mu - mu0).T

    def student_t(x, v, mu, cov):
        v = float(v)
        d = float(mu.shape[0])
        cov = np.mat(cov)
        print '!!!', (np.linalg.det(cov) ** -.5) / ((v*np.pi) ** (d/2)) 
        print x.shape, mu.shape, (x-mu).shape
        print ((x-mu)*cov.I*(x-mu).T)[0,0]
        print '~~~', math.pow(1 + 1./v * ((x-mu)*cov.I*(x-mu).T)[0,0], -(x+d)/2)
        return gamma(v/2 + d/2) / gamma(v/2) * \
               (np.linalg.det(cov) ** -.5) / ((v*np.pi) ** (d/2)) * \
               (1 + 1./v * (x-mu).T*cov.I*(x-mu))**(-(x+d)/2)

    return student_t(xi, vn-d+1, mun, covn*(kn+1) / (kn*(vn-d+1)))
    

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
        if it % 20 == 0:
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
