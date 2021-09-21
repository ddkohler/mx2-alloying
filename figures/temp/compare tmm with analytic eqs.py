import pathlib
import WrightTools as wt
import numpy as np
import matplotlib.pyplot as plt
import tmm_lib as lib
import optical_model as om


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


