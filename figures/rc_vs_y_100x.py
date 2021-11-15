import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt

import figlib as fl
import tmm_lib as lib

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
hw = np.linspace(1.6, 2.7, 201)
nmos2 = fl.n_mos2
nws2 = fl.n_ws2


def run(save):
    x100 =  wt.open(data_dir / "reflection_microspectroscopy" / "reflection.wt5")
    x20 = wt.open(data_dir / "reflection_microspectroscopy" / "reflection_20x.wt5")

    dx20 = x20.spectrum
    dx20.transform("energy", "ydist")
    dx20.contrast.clip(-0.35, 0.35)

    img = x100.image
    spectrum = x100.spectrum

    fig, gs = wt.artists.create_figure(
        width="double", nrows=4,
        cols=[1, 1, 1],
        aspects=[[[0,0], 0.1],[[1,0], 1],[[2,0], 0.2],[[3,0], 1]],
        hspace=0.1, wspace=0.3
    )

    ax0 = fig.add_subplot(gs[1,0])
    ax0.set_facecolor((0,0,0,0))
    ax0.pcolormesh(img, channel="signal", cmap="gist_gray")
    ax0.set_ylim(-30, 30)
    ax0.set_xlim(-30, 30)
    ax1 = fig.add_subplot(gs[1,1:3])
    # ax1.set_title(r"$x=0$")
    ax1.pcolormesh(spectrum, channel="contrast")
    ax1.vlines([1e7/532/8065.5], ymin=img.y.min(), ymax=img.y.max(), linestyle='-', color="g", linewidth=3, alpha=0.7)
    y = spectrum.y[:][0]
    valid = y > -23
    for x, linestyle in zip([spectrum.ares[:][0], spectrum.bres[:][0]], ["-", "-"]):
        ax1.plot(
            x[valid], y[valid],
            color="k", linewidth=1, alpha=0.3, linestyle=linestyle
        )
    ax0.vlines(
        [0], ymin=img.y.min(), ymax=img.y.max(), linestyle="--", color="k"
    )

    ax0.grid(axis="y", color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax1.grid(axis="y", color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax1.grid(axis="x", color="k", linestyle='-', alpha=0.3)
    wt.artists.set_ax_labels(ax0, xlabel=r"$x \ (\mu\mathsf{m})$", ylabel=r"$y \ (\mu\mathsf{m})$")
    wt.artists.set_ax_labels(ax1, xlabel=r"$\hbar\omega \ (\mathsf{eV})$")
    ax1.set_yticklabels(["" for i in ax1.get_yticklabels()])
    cax = fig.add_subplot(gs[0, 1:3])

    vmag = max(spectrum.contrast.min(), spectrum.contrast.max())
    wt.artists.plot_colorbar(
        cax,
        cmap="signed",
        ticks=np.linspace(-0.3, 0.3, 7),
        vlim=[-vmag, vmag],
        orientation="horizontal",
        decimals=1,
        label=r"$\left(R-R_0\right) / R_0$"
    )
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')

    ax2 = plt.subplot(gs[3,2])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    angle1 = 30 * np.pi / 180  # angle of rc trace from interface normal
    angle2 = 0 * np.pi / 180  # angle of pl trace from interface normal
    root = wt.open(data_dir / "ZYZ543.wt5")
    x0 = 0
    pl = root.PL.proc_PL.chop("energy", "y", at={"x":[x0, "um"]})[0]
    pl_max = np.array([pl.energy[i] for i in np.argmax(pl.intensity[:], axis=0)])
    ax2.plot(
        spectrum.y.points * np.cos(angle1),
        spectrum.ares.points,
        color="k", ls="-",
        label="RC"
    )
    ax2.plot(
        -(pl.y.points * np.cos(angle2) - 20),
        pl_max,
        color="k", ls=":",
        label="PPL"
    )
    ax2.plot(
        -(pl.y.points * np.cos(angle2) - 20),
        pl.intensity_energy_moment_1[0],
        color="k", ls="--"   ,     
        label=r"$\langle$PL$\rangle$"
    )
    ax2.set_ylim(1.8, 1.98)
    ax2.set_xlim(-20, 15)
    wt.artists.set_ax_labels(
        ax2,
        xlabel=r"$y \perp \ \left(\mathsf{\mu m}\right)$",
        ylabel=r"$\hbar \omega \ \left(\mathsf{eV}\right)$"
    )
    ax2.grid(ls=":", color="k")
    l = plt.legend(loc=6)
    l.set_alpha(0.8)

    ax3 = plt.subplot(gs[3, 0])
    ax4 = plt.subplot(gs[3, 1], sharey=ax3)
    wt.artists.corner_text(r"MoS$_2$, 0.46 NA", ax=ax3, background_alpha=.8, bbox=True, corner="LR")
    wt.artists.corner_text(r"WS$_2$, 0.46 NA", ax=ax4, background_alpha=.8, bbox=True, corner="LR")
    plt.yticks(visible=False)

    # chose specific points
    y_ws2 = dx20.split("ydist", [15, 20])[1].contrast[:].mean(axis=1)
    # y_mos2 = dx20.split("ydist", [31, 35])[1].contrast[:].mean(axis=1)
    y_mos2 = dx20.split("ydist", [-41, -39])[1].contrast[:].mean(axis=1)
    x = dx20.energy.points[:]

    blank = lib.FilmStack([fl.n_air, fl.n_fused_silica, fl.n_Si], [fl.d_SiO2])
    sample = lib.FilmStack([fl.n_air, nmos2, fl.n_fused_silica, fl.n_Si], [fl.d_mono, fl.d_SiO2])
    r_blank = blank.RT(hw)[0]
    r_sample = sample.RT(hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax3.plot(x, y_mos2, label=r"MoS$_2$", color="k", alpha=0.8)
    ax3.plot(hw, contrast, linewidth=2, linestyle=":", alpha=0.8, color="k", label=r"MoS$_2$ (theoretical)")

    blank = lib.FilmStack([fl.n_air, fl.n_fused_silica, fl.n_Si], [fl.d_SiO2])
    sample = lib.FilmStack([fl.n_air, nws2, fl.n_fused_silica, fl.n_Si], [fl.d_mono, fl.d_SiO2])
    r_blank = blank.RT(hw)[0]
    r_sample = sample.RT(hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    ax4.plot(x, y_ws2, label=r"WS$_2$", color="k", alpha=0.8)
    ax4.plot(hw, contrast, linewidth=2, linestyle=":", alpha=0.8, color="k", label=r"WS$_2$ (theoretical)")

    for ax in [ax3, ax4]:
        ax.grid(ls=":", c="k")
        ax.set_xlim(1.6, 2.65)
    wt.artists.set_ax_labels(
        ax3,
        xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$",
        ylabel=r"$(R-R_0) / R_0$",
    )
    wt.artists.set_ax_labels(
        ax4,
        xlabel=r"$\hbar\omega \ \left(\mathsf{eV}\right)$",
    )

    # from matplotlib.lines import Line2D
    # custom_lines = [
    #     Line2D([0], [0], color="k", ls=":"),
    #     Line2D([0], [0], color="k"),
    #     Line2D([0], [0], color="k", lw=4, alpha=0.5),
    # ]
    # ax4.legend(
    #     custom_lines,
    #     ['NA 0.95', 'NA 0.46', 'NA ~ 0 \n (theory)'],
    #     loc=[0.6, 0.1],
    # )
    wt.artists.corner_text("0.95 NA", ax=ax1, background_alpha=.8, bbox=True, corner="LR")

    for i, ax in enumerate([ax0, ax1, ax3, ax4, ax2]):
        wt.artists.corner_text(
            "abcde"[i],
            ax=ax,
            background_alpha=.8,
            bbox=True,
        )

    # save
    if save:
        p = "junction_reflection_contrast.png"
        p = here / p
        wt.artists.savefig(p)
    else:
        plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    run(save)

