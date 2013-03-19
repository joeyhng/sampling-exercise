import numpy as np
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
        print Xi
        print Xi.shape
        X.append(Xi)
    return np.vstack(X), centers, cov


def gibbs_sampling(X):
    pass


def visualize_data(X, centers, cov, label=None):
    print X.shape
    clf()
    if label is None:
        plot(X[:,0], X[:,1], '.')
        for i in range(len(centers)):
            error_ellipse(centers[i], cov[i])
        show()
    else:
        pass


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
    visualize_data(X, centers, cov)
    

if __name__ == '__main__':
    main()
