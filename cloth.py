import numpy as np


def createRandomCloth(n, center=(0.,0.,0.)):
    x = .5 * np.random.rand(n) + center[0]/2
    y = .5 * np.random.rand(n) + center[1]/2
    z = 1e-5 * np.random.rand(n)+center[2]
    return list(np.vstack([x, y, z]).T)


def createRectCloth(w, h, dw, dh, leftbtm=(0.,0.,0.)):
    points = []
    for i in range(w):
        for j in range(h):
            x = dw * i + leftbtm[0]
            y = dh * j + leftbtm[1]
            z = 1e-5*i*j + leftbtm[2]
            points.append([x,y,z])
    return points