import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import figlib as fl
import tmm_lib as lib


__here__ = pathlib.Path(__file__).resolve().parent
datapath = __here__.parent / "data" / "reflection_microspectroscopy"

hw = np.linspace(1.6, 2.7, 201)


def run1(save):
    root = wt.open(datapath / "reflection_20x.wt5")
    root.print_tree()

    lamp = root.lamp

    data = root.spectrum
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
    ax1.pcolormesh(data.split("energy", [2.5])[0], channel="contrast")
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
    wt.artists.set_ax_labels(
        ax0,
        xlabel=r"$\mathsf{x \ position \ (\mu m)}$",
        ylabel=r"$\mathsf{y \ position \ (\mu m)}$"
    )
    wt.artists.corner_text("a", ax=ax0)
    wt.artists.corner_text("b", ax=ax1)

    if save:
        wt.artists.savefig(__here__ / "reflection_contrast_20x.png")
    else:
        plt.show()

def run2(save):
    root = wt.open(datapath / "reflection_20x.wt5")
    data = root.spectrum
    data.transform("energy", "ydist")
    data.contrast.clip(-0.35, 0.35)

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

    y_ws2 = data.split("ydist", [15, 20])[1].contrast[:].mean(axis=1)
    y_mos2 = data.split("ydist", [31, 35])[1].contrast[:].mean(axis=1)
    x = data.energy.points[:]

    nmos2 = fl.n_mos2
    nws2 = fl.n_ws2

    # MoS2
    blank = lib.FilmStack([fl.n_air, fl.n_fused_silica, fl.n_Si], [fl.d_SiO2])
    sample = lib.FilmStack([fl.n_air, nmos2, fl.n_fused_silica, fl.n_Si], [fl.d_mono, fl.d_SiO2])
    r_blank = blank.RT(hw)[0]
    r_sample = sample.RT(hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax2.plot(data.energy.points[:], y_mos2, label=r"MoS$_2$", color="k")
    ax2.plot(hw, contrast, linewidth=4, alpha=0.5, color="k", label=r"MoS$_2$ (theoretical)")

    # WS2
    blank = lib.FilmStack([fl.n_air, fl.n_fused_silica, fl.n_Si], [fl.d_SiO2])
    sample = lib.FilmStack([fl.n_air, nws2, fl.n_fused_silica, fl.n_Si], [fl.d_mono, fl.d_SiO2])
    r_blank = blank.RT(hw)[0]
    r_sample = sample.RT(hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax3.plot(data.energy.points[:], y_ws2, label=r"WS$_2$", color="k")
    ax3.plot(hw, contrast, linewidth=4, alpha=0.5, color="k", label=r"WS$_2$ (theoretical)")

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
    if save:
        wt.artists.savefig(__here__ / "reflection_contrast_vs_na.png")
    else:
        plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    run1(save)
    # run2(save)
