import numpy as np


def sin1d(left, right, a, n=100):
    x = np.random.uniform(left, right, (n, 1))
    y = a * np.sin(x)
    return x, y


def cosc(x):
    return 2 * np.square(2 * np.cos(x)**2 - 1) - 1



def sinc(x):
    return np.sin(4 * x) #/ np.pi


def cosc_data(left, right, a, n):
    x = np.random.uniform(left, right, (n, 1))
    y = cosc(x) * a
    return x, y

