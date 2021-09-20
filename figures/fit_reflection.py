import pathlib
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt
import tmm_lib as lib
import optical_model as om


here = pathlib.Path(__file__).resolve().parent
data = here.parent / "data" / "reflection_microspectroscopy" / "reflection.wt5"
root = wt.open(data)

d = root["spectrum"]
hw = np.linspace(1.6, 2.7, 201)
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
    r"https://refractiveindex.info/database/data/main/WS2/Hsu-1L.yml", 
    skip_header=10, skip_footer=5
)

n_Si = from_refractiveindex_info(
    r"https://refractiveindex.info/database/data/main/Si/Schinke.yml",
    skip_header=11, skip_footer=2
)

n_fused_silica = lib.n_fused_silica
n_air = lib.n_air

wt.artists.apply_rcparams(kind="publication")


# from Darien's fitting ---------------------------------------------------------------------------


def L(E, E0, G, A):
    return A * G / (E0 - E - 1j * G)


def n_semi_1D(E0s, Gs, As, e1_dc, offset):
    def f(E):
        Enew = E[:, None]
        E0new = E0s[None, :]
        Gsnew = Gs[None, :]
        Asnew = As[None, :]
        out = L(Enew, E0new, Gsnew, Asnew) + offset
        out += e1_dc / Enew**2
        return np.sum(out, axis=-1)**0.5
    return f


# definition of dielectric spectrum MoS2
offset_mos2 = 9
E0s_mos2 = np.array([1.83, 2.04, 2.85])
Gs_mos2 = np.array([.02, .06, .23])
As_mos2 = np.array([8, 10, 10])  # 2020-01-29
nmos2 = n_semi_1D(E0s_mos2, Gs_mos2, As_mos2, 0, offset_mos2)
# definition of dielectric spectrum WS2
offset_ws2 = 12  # 2020-01-29
E0s_ws2 = np.array([1.94 - .005, 2.38, 2.85])
Gs_ws2 = np.array([.03 + .005, .06, .2])
Gs_ws2 = np.array([.045, .07, .2])  # 2020-01-29
As_ws2 = np.array([8, 4, 20])
As_ws2 = np.array([8, 5, 20])  # 2020-01-29
# nws2 = n_semi_1D(E0s_ws2, Gs_ws2, As_ws2, offset_ws2)

dsamp = 0.7
dsio2 = 260  # 2020-01-30
cws2 = .2
cmos2 = .30
crest = 1 - cws2 - cmos2

if False:  # checking agreement between the functions we use (I use TMM, Darien has explicit equation)
    # everything here looks great for agreement--I initially got tripped up because I ordered the layers in reverse
    wl = 1e7 / (hw * 8065.5)
    Rsample = np.abs(om.three_layer_complexR(
        wl, n_air(hw), nmos2(hw), n_fused_silica(hw), n_Si(hw), dsamp, dsio2
    ))**2
    Rsample_sio2 = np.abs(om.three_layer_complexR(
        wl, n_air(hw), nmos2(hw), n_fused_silica(hw), n_fused_silica(hw), dsamp, dsio2
    ))**2
    Rsio2 = np.abs(om.three_layer_complexR(
        wl, n_air(hw), n_air(hw), n_fused_silica(hw), n_fused_silica(hw), dsamp, dsio2
    ))**2
    Rsisio2 = np.abs(om.three_layer_complexR(
        wl, n_air(hw), n_air(hw), n_fused_silica(hw), n_Si(hw), dsamp, dsio2
    ))**2


    # my simulation
    d2 = dsio2 * 1e-7
    blank = lib.FilmStack([n_air, n_fused_silica, n_Si], [d2])
    sample = lib.FilmStack([n_air, nmos2, n_fused_silica, n_Si], [d_mono, d2])
    r_blank = blank.RT(hw)[0]
    r_sample = sample.RT(hw)[0]  # NOTE: my "t backside" matches darien's reflectivity
 
    plt.figure()
    plt.plot(hw, (Rsample - Rsisio2) / Rsisio2, color="r", linestyle=":")
    plt.plot(hw, (r_sample - r_blank) / r_blank, color="b", alpha=0.3, linewidth=5)
    # plt.plot(hw, r_blank, color="k")
    # plt.plot(hw, (Rsample - Rsisio2) / Rsisio2)
    # plt.plot(hw, (r_sample - r_blank) / r_blank)


# -------------------------------------------------------------------------------------------------


if False:
    fig, gs = wt.artists.create_figure(width="dissertation")
    ax = plt.subplot(gs[0])

    ds = np.linspace(220e-7, 340e-7, 7)
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
    # d2 = 75e-7
    fig, gs = wt.artists.create_figure(width="dissertation", default_aspect=0.5)
    ax = plt.subplot(gs[0])
    # plt.title(f"Reflection Contrast vs. SIO2 Thickness ")
    for i, d2i in enumerate(np.linspace(240, 340, 6)):
        # nmos2 = n_semi_1D(E0s_mos2, Gs_mos2, As_mos2, 0, offset_mos2)
        def nmos2(hw):
            return n_mos2_ml(hw + 0.04)
        label = str(d2i)
        if i==0:
            label += " nm"
        blank = lib.FilmStack([n_air, n_fused_silica, n_Si], [d2i * 1e-7])
        sample = lib.FilmStack([n_air, nmos2, n_fused_silica, n_Si], [d_mono, d2i * 1e-7])
        r_blank = blank.RT(hw)[0]
        r_sample = sample.RT(hw)[0]
        contrast = (r_sample - r_blank) / r_blank
        ax.plot(hw, contrast, linewidth=3, alpha=0.5, label=label)

    for y in [-15]:
        speci = d.chop("wm", at={"y":[y, "um"]})[0]
        # speci.print_tree()
        ax.plot(
            speci.wm[:], speci.contrast[:],
            color="k", label="experiment \n" + r"($x=0 \ \mu \mathsf{m}, y=-16 \ \mu \mathsf{m}$)"
        )
    l = ax.legend(fontsize=12, framealpha=0.7)
    plt.ylabel(r"$(R - R_0) / R_0$", fontsize=18)
    plt.xlabel(r"$\hbar \omega \ \left(\mathsf{eV} \right)$", fontsize=18)
    plt.grid()
    wt.artists.savefig(here / "constrast vs SiO2 thickness.png")

# plt.show()
