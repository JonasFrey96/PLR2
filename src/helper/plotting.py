import k3d
import numpy as np


def plot_points(x, point_size=0.005, c='g'):
    """
    x: point_nr,3
    """
    if c == 'b':
        k = 245
    elif c == 'g':
        k = 25811000
    elif c == 'r':
        k = 11801000
    elif c == 'black':
        k = 2580
    else:
        k = 2580
    colors = np.ones(x.shape[0]) * k
    plot = k3d.plot(name='points')
    print(colors.shape)
    plt_points = k3d.points(x, colors.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


def plot_two_pc(x, y, point_size=0.005, c1='g', c2='r'):
    if c1 == 'b':
        k = 245
    elif c1 == 'g':
        k = 25811000
    elif c1 == 'r':
        k = 11801000
    elif c1 == 'black':
        k = 2580
    else:
        k = 2580

    if c2 == 'b':
        k2 = 245
    elif c2 == 'g':
        k2 = 25811000
    elif c2 == 'r':
        k2 = 11801000
    elif c2 == 'black':
        k2 = 2580
    else:
        k2 = 2580

    col1 = np.ones(x.shape[0]) * k
    col2 = np.ones(y.shape[0]) * k2
    plot = k3d.plot(name='points')
    plt_points = k3d.points(x, col1.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points = k3d.points(y, col2.astype(np.uint32), point_size=point_size)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()
