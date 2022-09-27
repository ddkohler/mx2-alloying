import numpy as np

def gauss(x, p):
    a, r, w = p
    s = w / 2.35
    return a * np.exp(-(x-r)**2 / (2**0.5 * s)**2)


def lorentzian(x, p):
    a, r, w = p
    return a * w**2 / ((x-r)**2 + w**2)


def two_res(x, p):
    a1, a2, r1, r2, w1, w2 = p
    out = gauss(x, [a1, r1, w1])
    out += gauss(x, [a2, r2, w2])
    return out


def fom(p, w, y):
    fit = two_res(w, p)
    return y - fit
