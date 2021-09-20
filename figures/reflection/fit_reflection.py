import pathlib
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt
import tmm_lib as lib


here = pathlib.Path(__file__).parent
# data = here.parent / "data" / "reflection_microspectroscopy" / "reflection.wt5"
# root = wt.open(data)

# d = root["spectrum"]
hw = np.linspace(1.5, 2.6, 201)
d_mono = 7e-8  # cm


def from_refractiveindex_info(url, **kwargs):
    arr = np.genfromtxt(url, **kwargs, unpack=True)
    from scipy.interpolate import interp1d
    func = interp1d(1e4 / arr[0] / 8065.5, arr[1] + 1j * arr[2], kind='quadratic')
    return func


n_mos2_ml = from_refractiveindex_info(
    r"https://refractiveindex.info/database/data/main/MoS2/Islam-1L.yml",
    skip_header=10, skip_footer=5,
)




n_ws2_ml = from_refractiveindex_info(
    # r"https://refractiveindex.info/database/data/main/WS2/Hsu-1L.yml", 
    r"https://refractiveindex.info/database/data/main/WS2/Ermolaev.yml",
    skip_header=11, skip_footer=5
)

n_Si = from_refractiveindex_info(
    r"https://refractiveindex.info/database/data/main/Si/Schinke.yml",
    skip_header=11, skip_footer=2
)

n_fused_silica = lib.n_fused_silica
n_air = lib.n_air


# from Darien's fitting ---------------------------------------------------------------------------


def L(E, E0, G, A):
    return A * G / (E0 - E - 1j * G)


def n_semi_1D(E0s, Gs, As, e1_dc, offset):
    def f(E):
        Enew = E[:, None]
        E0new = E0s[None, :]
        Gsnew = Gs[None, :]
        Asnew = As[None, :]
        out = L(Enew, E0new, Gsnew, Asnew)
        out += e1_dc / Enew**2
        return (np.sum(out, axis=-1) + offset)**0.5
    return f


# definition of dielectric spectrum MoS2
offset_mos2 = 5
E0s_mos2 = np.array([1.85, 2.05, 2.85])
Gs_mos2 = np.array([.04, .08, .23])
As_mos2 = np.array([10, 10, 30])  # 2020-09-13
# nmos2 = n_semi_1D(E0s_mos2, Gs_mos2, As_mos2, offset_mos2)
# definition of dielectric spectrum WS2
offset_ws2 = 15  # 2020-01-29
E0s_ws2 = np.array([1.96, 2.3, 2.85])
Gs_ws2 = np.array([.02, .07, .2])  # 2020-01-29
As_ws2 = np.array([30, 7, 20])  # 2020-09-13
# nws2 = n_semi_1D(E0s_ws2, Gs_ws2, As_ws2, offset_ws2)

dsamp = 0.7
dsio2 = 260  # 2020-01-30
cws2 = .2
cmos2 = .30
crest = 1 - cws2 - cmos2


# -------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    if False:
        fig, gs = wt.artists.create_figure()
        ax = plt.subplot(gs[0])

        ds = np.linspace(200e-7, 400e-7, 6)
        for i, d2 in enumerate(ds):
            label = str(int(d2 * 1e7))
            if i==0:
                label += " nm"
            blank = lib.FilmStack([n_Si, n_fused_silica, n_air], [d2])
            sample = lib.FilmStack([n_Si, n_fused_silica, nmos2, n_air], [d2, d_mono])
            r_blank = blank.RT(hw)[0]
            r_sample = sample.RT(hw)[0]
            contrast = (r_sample - r_blank) / r_blank
            ax.plot(hw, contrast, label=label)
        ax.legend(fontsize=10)
        plt.grid()


    if False:
        d.level("contrast", 1, 60)
        d.level("contrast", 0, -1)
        d.contrast *= -1
        d.contrast.signed = False
        # d = d.split("y", [-6, 6])[1]
        out = wt.artists.interact2D(d, channel="contrast")

    # d.print_tree()

    if False:  # checking that Darien's n_Si is consistent with mine; agreement is great
        lit = wt.open(here.parent / "data" / "literature_refractive_index.wt5")
        nsi = lit.Si
        nsi.print_tree()
        plt.figure()
        plt.plot(nsi.energy[:], nsi.n[:])


    if True:
        plt.figure()
        plt.title(f"reflection contrast vs d2")
        for i, d2 in enumerate(np.linspace(260e-7, 300e-7, 7)):
            nmos2 = n_semi_1D(E0s_mos2, Gs_mos2, As_mos2, 9, 1)
            label = str(d2)
            blank = lib.FilmStack([n_air, n_fused_silica, n_Si], [d2])
            sample = lib.FilmStack([n_air, nmos2, n_fused_silica, n_Si], [d_mono, d2])
            r_blank = blank.RT(hw)[0]
            r_sample = sample.RT(hw)[0]
            contrast = (r_sample - r_blank) / r_blank
            plt.plot(hw, contrast, linewidth=3, alpha=0.5, label=label)

        plt.legend(fontsize=10)
        # for y in np.linspace(-5, -20, 5):
        #     speci = d.chop("wm", at={"y":[y, "um"]})[0]
        #     speci.print_tree()
        #     plt.plot(
        #         speci.wm[:], speci.contrast[:],
        #         color="k", label=r"exp. (x=0 $\mu$m, y=-16 $\mu$m"
        #     )
        plt.ylabel("reflection contrast")
        plt.xlabel("energy (eV)")
        plt.grid()

    plt.show()
