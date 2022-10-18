"""
=============================================
Generate polygons to fill under 3D line graph
=============================================

Demonstrate how to create polygons which fill the space under a line
graph. In this example polygons are semi-transparent, creating a sort
of 'jagged stained glass' effect.
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import random
import hackathon_em as em


def waterfallThree():
    # Get the data (csv file is hosted on the web)
    url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
    data = pd.read_csv(url)

    # Transform it to a long format
    df = data.unstack().reset_index()
    df.columns = ["X", "Y", "Z"]

    # And transform the old column name in something numeric
    df['X'] = pd.Categorical(df['X'])
    df['X'] = df['X'].cat.codes

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis,
                    linewidth=0.2)
    plt.show()

    # to Add a color bar which maps values to colors.
    surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis,
                           linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Rotate it
    ax.view_init(30, 45)
    plt.show()

    # Other palette
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
    plt.show()



def generateDataTwo():
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X ** 2 + Y ** 2)

    # x = np.linspace(0, 1, 100)
    # y = em.beta.pdf(x, a=10, b=100)
    # X, Y = np.meshgrid(x, y)
    # z = range(0, len(x))
    # Z = np.array(z)

    return X, Y, Z


def waterfallTwo(X,Y,Z):
    '''
    Make a waterfall plot
    Input:
        fig,ax : matplotlib figure and axes to populate
        Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
        X,Y : n,m array
    '''
    # Generate waterfall plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set normalization to the same values for all plots
    norm = plt.Normalize(Z.min().min(), Z.max().max())
    # Check sizes to loop always over the smallest dimension
    n,m = Z.shape
    if n>m:
        X=X.T; Y=Y.T; Z=Z.T
        m,n = n,m

    for j in range(n):
        # reshape the X,Z into pairs
        points = np.array([X[j,:], Z[j,:]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        # Set the values used for colormapping
        lc.set_array((Z[j,1:]+Z[j,:-1])/2)
        lc.set_linewidth(2) # set linewidth a little larger to see properly the colormap variation
        line = ax.add_collection3d(lc,zs=(Y[j,1:]+Y[j,:-1])/2, zdir='y') # add line to axes

    fig.colorbar(lc) # add colorbar, as the normalization is the same for all, it doesent matter which of the lc objects we use

    ax.set_xlabel('X')
    ax.set_xlim3d(-2, 2)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-2, 2)
    ax.set_zlabel('Z')
    ax.set_zlim3d(-1, 1)
    plt.show()


def waterfallOne(verts, zs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # xs = np.arange(0, 10, 0.4)
    # verts = []
    # zs = [0.0, 1.0, 2.0, 3.0]
    # for z in zs:
    #     ys = np.random.rand(len(xs))
    #     ys[0], ys[-1] = 0, 0
    #     verts.append(list(zip(xs, ys)))

    poly = PolyCollection(verts, facecolors=[mcolors.to_rgba('r', alpha=0.6),
                                             mcolors.to_rgba('g', alpha=0.6),
                                             mcolors.to_rgba('b', alpha=0.6),
                                             mcolors.to_rgba('y', alpha=0.6)])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 4)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 1)

    plt.show()

def generateDataOne():
    xs = np.linspace(0, 10, 100)
    verts = []
    zs = range(0, len(xs))
    for z in zs:
        ys = em.beta.pdf(xs, a=10, b=100)
        ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))

    return verts, zs

def main():
    # x = np.linspace(0, 1, 50)
    # Z = np.zeros([100, 50])
    # for i in range(0, 100):
    #     alpha = random.randint(1, 1 + i)
    #     beta = random.randint(1, 10 - i)
    #     y = em.beta.pdf(x, a=alpha, b=beta)
    #     Z[0, i] = y
    # N = 100
    # xf = np.linspace(0.0, 1.0, N // 2)
    # x, y = (xf, np.arange(N))
    # X, Y = np.meshgrid(x, y)
    # Axes3D.plot_wireframe(X, Y, Z, rstride=1, cstride=len(xf), lw=.5,alpha=0.5)




    # verts, zs = generateDataOne()
    # waterfallOne(verts, zs)
    #
    X, Y, Z = generateDataTwo()
    # waterfallTwo(X, Y, Z)

    # waterfallThree()


if __name__ == '__main__':
    main()
