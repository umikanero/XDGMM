# coding=UTF-8
# This Python file uses the following encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
from matplotlib.pyplot import MultipleLocator
import os
import astropy.coordinates as apycoords

def visualize_3d_gmm(points, w, mu, stdev, index, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: gmm.covariances_ (assuming diagonal covariance matrix)
    Output:
        None
    '''
    points = points.astype('float64')
    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    plt.grid()
    #plt.gca().set_aspect("equal")
    for i in range(n_gaussians):
        covariances = stdev[i][:3, :3]
        filename = 'Test_XDGMM'
        v, u = np.linalg.eigh(covariances)
        #r = 2. * np.sqrt(2.) * np.sqrt(v)
        r = np.sqrt(v)
        # print(mu)
        data = points[np.where(index == i)]
        # inner, outer = find_fraction(data, center=mu[:3, i], r=r, rotation=u)
        # print(inner / (inner + outer))
        plot_sphere(w=w[i], center=mu[:3, i], r=r, rotation=u, ax=axes)
        #[:, i]取所有行（即三个维度）的第i个数据

    for n in range(n_gaussians):
        data = points[np.where(index == n)]
        plt.set_cmap('Set1')
        colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
        print(data.shape)
        axes.scatter(data[:, 0], data[:, 1], data[:, 2], s = 2.0, alpha = 0.5, color = colors[n])

    plt.title(filename)
    axes.set_xlabel('X /Mpc')
    axes.set_ylabel('Y /Mpc')
    axes.set_zlabel('Z /MPc')
    axes.set_zlim3d(-5, 5)
    axes.set_xlim3d(-5, 5)
    axes.set_ylim3d(-5, 5)
    # x_major_locator = MultipleLocator(50)
    # # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(50)
    # # 把y轴的刻度间隔设置为10，并存在变量里
    # axes.xaxis.set_major_locator(x_major_locator)
    # # 把x轴的主刻度设置为1的倍数
    # axes.yaxis.set_major_locator(y_major_locator)
    # axes.view_init(30, 60)
    plt.savefig(filename, dpi=100, format='png')
    plt.show()


def plot_sphere(w=0, center=[0,0,0], r=[1, 1, 1], rotation=[1,1,1], ax=None):
    '''
        plot a sphere surface
        Input:
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
                    是椭球的分辨率
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 30)   #np.linspace 取等差数列
    v = np.linspace(0, np.pi, 30)
    x = r[0] * np.outer(np.cos(u), np.sin(v))
    y = r[1] * np.outer(np.sin(u), np.sin(v))
    z = r[2] * np.outer(np.ones(np.size(u)), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            #[x[i, j], y[i, j], z[i, j]] = [x[i, j], y[i, j], z[i, j]] + center #spherical专用
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    ax.plot_surface(x, y, z, alpha=0.6)

    return ax

def find_fraction(points, center=[0,0,0], r=[1, 1, 1], rotation=[1,1,1]):
    inner = 0.0
    outer = 0.0
    x = points[:,0] - center[0]
    y = points[:,1] - center[1]
    z = points[:,2] - center[2]
    r = 3 * r   # 3 sigma球
    for j in range(len(x)):
        [x[j], y[j], z[j]] = np.dot([x[j], y[j], z[j]], np.linalg.inv(rotation))
    for i in range(points.shape[0]):
        distance = np.square(x[i]/r[0]) + np.square(y[i]/r[1]) + np.square(z[i]/r[2])
        if distance > 1.0:
            outer +=1.0
        elif distance < 1.0:
            inner +=1.0

    return inner, outer
