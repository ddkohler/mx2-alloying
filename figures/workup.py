import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import fit_reflection as fr
import tmm_lib as lib


__here__ = pathlib.Path(__file__).resolve().parent
datapath = __here__.parent / "data" / "reflection_microspectroscopy"


def run(all_plot, save):
    root = wt.open(datapath / "reflection_20x.wt5")
    root.print_tree()

    lamp = root.lamp

    data = root.spectrum
    data.smooth((10,0))

    substrate_low = data.split("yindex", [900])[0].signal[:].mean(axis=1)[:, None]
    substrate_high = data.split("yindex", [1250])[1].signal[:].mean(axis=1)[:, None]
    # interpolate between top and bottom substrate spectra 
    z = data.yindex[:].copy()
    s = (z - 900) / 350
    s[s>1] = 1
    s[s<0] = 0
    substrate = (1-s) * substrate_low + s * substrate_high

    data.create_channel(
        "contrast", values=(data.signal[:] - substrate) / substrate, signed=True
    )
    data.transform("energy", "ydist")
    data.contrast.clip(-0.35, 0.35)

    # fig 1: 20x results
    fig, gs = wt.artists.create_figure(width="dissertation", nrows=1, cols=[1,1,"cbar"])

    ax0 = plt.subplot(gs[0, 0])
    ax0.set_title("image (20x) \n ")
    ax0.pcolormesh(lamp, cmap="gist_gray")
    ax0.grid()

    ax1 = plt.subplot(gs[0, 1])
    ax1.set_title("reflection contrast \n (y=0)")
    ax1.pcolormesh(data, channel="contrast")
    plt.yticks(visible=False)
    ax1.grid()

    cax = plt.subplot(gs[0, 2])
    wt.artists.plot_colorbar(
        cax,
        cmap="signed",
        ticks=np.linspace(-0.3, 0.3, 7),
        clim=[-0.35, 0.35],
        label=r"$(R-R_0) / R_0$"
    )
    
    wt.artists.set_ax_labels(ax1, xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$")
    wt.artists.set_ax_labels(ax0, xlabel=r"$\mathsf{x \ position \ (\mu m)}$", ylabel=r"$\mathsf{y \ position \ (\mu m)}$")
    wt.artists.corner_text("a", ax=ax0)
    wt.artists.corner_text("b", ax=ax1)
    wt.artists.savefig(__here__ / "reflection_contrast_20x.png")

    # fig 2: comparison of different NAs
    fig, gs = wt.artists.create_figure(width="dissertation", nrows=1, cols=[1, 1, 0.3])

    ax2 = plt.subplot(gs[0, 0])
    ax3 = plt.subplot(gs[0, 1], sharey=ax2)
    plt.yticks(visible=False)
    ax2.set_title(r"MoS$_2$")
    ax3.set_title(r"WS$_2$")
    ax2.grid()
    ax3.grid()
    wt.artists.set_ax_labels(
        ax2,
        xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$",
        ylabel=r"$(R-R_0) / R_0$",
    )
    wt.artists.set_ax_labels(
        ax3,
        xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$",
    )

    # ax2.set_title("Comparison with Fresnel Effects")
    y_ws2 = data.split("ydist", [15, 20])[1].contrast[:].mean(axis=1)
    y_mos2 = data.split("ydist", [31, 35])[1].contrast[:].mean(axis=1)
    x = data.energy.points[:]

    d2 = 298e-7

    if True:  # apply offsets to MX2 optical constants
        def nmos2(hw):
            return fr.n_mos2_ml(hw + 0.04)
        def nws2(hw):
            return fr.n_ws2_ml(hw + 0.07)
    else:
        nmos2 = fr.n_mos2_ml
        nws2 = fr.n_ws2_ml

    # MoS2
    blank = lib.FilmStack([fr.n_air, fr.n_fused_silica, fr.n_Si], [d2])
    sample = lib.FilmStack([fr.n_air, nmos2, fr.n_fused_silica, fr.n_Si], [fr.d_mono, d2])
    r_blank = blank.RT(fr.hw)[0]
    r_sample = sample.RT(fr.hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax2.plot(data.energy.points[:], y_mos2, label=r"MoS$_2$", color="k")
    ax2.plot(fr.hw, contrast, linewidth=4, alpha=0.5, color="k", label=r"MoS$_2$ (theoretical)")

    # WS2
    blank = lib.FilmStack([fr.n_air, fr.n_fused_silica, fr.n_Si], [d2])
    sample = lib.FilmStack([fr.n_air, nws2, fr.n_fused_silica, fr.n_Si], [fr.d_mono, d2])
    r_blank = blank.RT(fr.hw)[0]
    r_sample = sample.RT(fr.hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax3.plot(data.energy.points[:], y_ws2, label=r"WS$_2$", color="k")
    ax3.plot(fr.hw, contrast, linewidth=4, alpha=0.5, color="k", label=r"WS$_2$ (theoretical)")

    # 100x
    x100 = wt.open(datapath / "reflection.wt5")
    rc2 = x100.spectrum
    y_ws2_100x = rc2.split("y", [15, 20])[1].contrast[:].mean(axis=1)
    y_mos2_100x = rc2.split("y", [-20, -15])[1].contrast[:].mean(axis=1)
    ax2.plot(rc2.wm.points[:], y_mos2_100x, label=r"MoS$_2$ (100x)", color="k", linestyle=":")
    ax3.plot(rc2.wm.points[:], y_ws2_100x, label=r"WS$_2$ (100x)", color="k", linestyle=":")

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="k", ls=":"),
        Line2D([0], [0], color="k"),
        Line2D([0], [0], color="k", lw=4, alpha=0.5),
    ]

    ax3.legend(custom_lines, ['NA 0.95', 'NA 0.46', 'NA ~ 0 \n (theory)'], fontsize=18, loc=[1.1, 0.5])
    wt.artists.corner_text("a", ax=ax2)
    wt.artists.corner_text("b", ax=ax3)
    wt.artists.savefig(__here__ / "reflection_contrast_vs_na.png")


if __name__ == "__main__":
    run(True, True)
