import numpy as np
import matplotlib.pyplot as plt
import WrightTools as wt
import pathlib
import matplotlib as mpl
from scipy.interpolate import interp1d


def complexn_from_d(E, d, ncomplex=True):
    if ncomplex:
        nout = interp1d(d.energy.points, d.n.points, kind="cubic")(E)
        kout = interp1d(d.energy.points, d.k.points, kind="cubic")(E)
        kout[kout < 0] = 0  # removes accidental gain
        return nout + 1j * kout
    else:
        nout = interp1d(d.energy.points, d.n.points, kind="cubic")(E)
        return nout + 1j * 0


def r_m1(n_m0, n_m1):
    return (n_m0 - n_m1) / (n_m0 + n_m1)


def delta_m(wl_vac, n_m, d_m):
    # wl and thickness must have same units.
    return 4 * np.pi * n_m * d_m / wl_vac


def R_three_layer_three_comp(wl_vac, n0, n11, n12, n13, n2, n3, d1, d2, c1, c2, c3):
    r1 = c1 * three_layer_complexR(wl_vac, n0, n11, n2, n3, d1, d2)
    r2 = c2 * three_layer_complexR(wl_vac, n0, n12, n2, n3, d1, d2)
    r3 = c3 * three_layer_complexR(wl_vac, n0, n13, n2, n3, d1, d2)
    return np.abs(r1 + r2 + r3) ** 2


def three_layer_complexR(wl_vac, n0, n1, n2, n3, d1, d2):
    r1 = r_m1(n0, n1)
    r2 = r_m1(n1, n2)
    r3 = r_m1(n2, n3)
    del1 = delta_m(wl_vac, n1, d1)
    del2 = delta_m(wl_vac, n2, d2)
    # these equations differ from Anders in that all 1j --> -1j
    numerator = (
        r1
        + r2 * np.exp(1j * del1)
        + r3 * np.exp(1j * (del1 + del2))
        + r1 * r2 * r3 * np.exp(1j * del2)
    )
    denominator = (
        1
        + r1 * r2 * np.exp(1j * del1)
        + r1 * r3 * np.exp(1j * (del1 + del2))
        + r2 * r3 * np.exp(1j * del2)
    )
    return numerator / denominator


def R_three_layer(wl_vac, n0, n1, n2, n3, d1, d2):
    out = three_layer_complexR(wl_vac, n0, n1, n2, n3, d1, d2)
    return np.abs(out) ** 2


def E_to_wl_nm(E):
    # E is in eV
    c = 2.9979e8 * 1e9  # nm/s
    h = 4.135668e-15  # eV s
    return h * c / E


def L(E, E0, G, A):
    return A * np.sqrt(G / np.pi) / (E0 - E - 1j * G)


def e_semi_1D(E, E0s, Gs, As):
    Enew = E[:, None]
    E0new = E0s[None, :]
    Gsnew = Gs[None, :]
    Asnew = As[None, :]
    out = L(Enew, E0new, Gsnew, Asnew)
    return np.sum(out, axis=-1)
