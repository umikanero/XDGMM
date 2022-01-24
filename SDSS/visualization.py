# coding=UTF-8
# This Python file uses the following encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os
from matplotlib.pyplot import MultipleLocator

def visualize_3d_gmm(points, w, mu, stdev, type, index, export=True):
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

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    #axes.set_xlim([-1, 1])
    #axes.set_ylim([-1, 1])
    #axes.set_zlim([-1, 1])
    plt.grid()
    #plt.gca().set_aspect("equal")
    for i in range(n_gaussians):
        if type == 'full':
            covariances = stdev[i][:3, :3]
            filename = 'SDSS full'
        elif type == 'tied':
            covariances = stdev[:3, :3]
            filename = 'VIPERS_GMM_W4a1_tied'
        elif type == 'diag':
            covariances = np.diag(stdev[i][:3])
            filename = 'VIPERS_GMM_W4a1_diag'
        elif type == 'spherical':
            covariances = np.eye(n_gaussians) * stdev[i]
            filename = 'VIPERS_GMM_W4a1_spherical'
        v, u = np.linalg.eigh(covariances)
        #r = 2. * np.sqrt(2.) * np.sqrt(v)
        r = np.sqrt(v)
        plot_sphere(w=w[i], center=mu[:3, i], r=r, rotation=u, ax=axes)
        #[:, i]取所有行（即三个维度）的第i个数据

    # for n in range(n_gaussians):
    #     data = points[np.where(index == n)]
    #     plt.set_cmap('Set1')
    #     colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    #     axes.scatter(data[:, 0], data[:, 1], data[:, 2], s = 1.0, alpha = 0.8, color = colors[n])

    plt.title(filename)
    axes.set_xlabel('X /Mpc')
    axes.set_ylabel('Y /Mpc')
    axes.set_zlabel('Z /MPc')
    x_major_locator = MultipleLocator(20)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(20)
    z_major_locator = MultipleLocator(20)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    ax.zaxis.set_major_locator(z_major_locator)
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
            # [x[i, j], y[i, j], z[i, j]] = [x[i, j], y[i, j], z[i, j]] + center #spherical专用
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    ax.plot_surface(x, y, z, alpha=0.1)

    return ax