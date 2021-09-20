"""simulate RPP stack transmission and reflectance
"""

import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt
import os
import lib

__here__ = os.path.abspath(os.path.dirname(__file__))
E = np.linspace(1.5, 3.0, 101)


# --- refractive indices --------------------------------------------------------------------------


def from_refractiveindex_info(url, **kwargs):
    arr = np.genfromtxt(url, **kwargs, unpack=True)
    from scipy.interpolate import interp1d
    func = interp1d(1e4 / arr[0] / 8065.5, arr[1] + 1j * arr[2], kind='quadratic')
    return func


n_ws2_ml = from_refractiveindex_info(
    r"https://refractiveindex.info/database/data/main/WS2/Hsu-1L.yml", 
    # r"https://refractiveindex.info/database/data/main/WS2/Ermolaev.yml", # CVD mono ellipsometry
    # r"https://refractiveindex.info/database/data/main/WS2/Jung.yml",
    skip_header=10, skip_footer=5
)
n_ws2 = from_refractiveindex_info(
    r"https://refractiveindex.info/database/data/main/WS2/Hsu-3L.yml",
    skip_header=10, skip_footer=5
)
n_Si = from_refractiveindex_info(
    r"https://refractiveindex.info/database/data/main/Si/Schinke.yml",
    skip_header=11, skip_footer=2
)
n_fused_silica = lib.n_fused_silica
n_air = lib.n_air

# --- thickness dependence ------------------------------------------------------------------------
# --- vary WS2 thickness 
if True:
    d2 = 350e-7  # thickness of fused silica layer (cm)
    glass_blank = lib.FilmStack([n_fused_silica, n_air], [0])
    protected_si_blank = lib.FilmStack([n_Si, n_fused_silica, n_air], [d2])
    E = np.linspace(1.85, 2.55)

    R_glass = glass_blank.RT(E)[0]
    R_prot_Si = protected_si_blank.RT(E)[0]

    fig, gs = wt.artists.create_figure(width="double", cols=[1, 1])
    ax0 = plt.subplot(gs[0])
    plt.grid(b=True)
    ax1 = plt.subplot(gs[1], sharey=ax0)
    plt.yticks(visible=False)
    plt.grid(b=True)
    ax0.set_title("glass")
    ax1.set_title("prot. Si ({0} nm)".format(d2*1e7))
    ax0.set_ylabel("contrast")
    ax0.set_xlabel("E (eV)")
    ax1.set_xlabel("E (eV)")

    ds = np.linspace(10, 70, 8) * 1e-7
    # ds[0] = 0.75e-7
    R_glass_mx2 = np.empty((ds.size, E.size), dtype=float)
    R_prot_Si_mx2 = np.empty((ds.size, E.size), dtype=float)

    for i, di in enumerate(ds):
        glass_mx2 = lib.FilmStack([n_fused_silica, n_ws2, n_air], [di])
        protected_si_mx2 = lib.FilmStack([n_Si, n_fused_silica, n_ws2, n_air], [d2, di])

        R_glass_mx2[i] = glass_mx2.RT(E)[0]
        R_prot_Si_mx2[i] = protected_si_mx2.RT(E)[0]

        label = str(int(di * 1e7))
        if i==0:
            label += " nm"
        for sample, blank, ax in [
            [R_glass_mx2[i], R_glass, ax0],
            [R_prot_Si_mx2[i], R_prot_Si, ax1]
        ]:
            contrast = (sample - blank) / (sample + blank)
            ax.plot(E, contrast, color=plt.cm.rainbow_r(i/ds.size), label=label)
        ax1.legend(fontsize=10)
    wt.artists.savefig(os.path.join(__here__, "thickness dependence.png"), facecolor="white")
