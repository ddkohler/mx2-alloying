import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import figlib as fl
import tmm_lib as lib
import numpy as np


grid_kwargs = fl.grid_kwargs
nmos2 = fl.n_mos2
nws2 = fl.n_ws2
hw = np.linspace(1.6, 2.7, 201)

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

ev_label = r"$\hbar\omega \ \left(\mathsf{eV}\right)$"


def sim_contrast(n_tmdc, d_tmdc, hw):
    blank = lib.FilmStack([fl.n_air, fl.n_fused_silica, fl.n_Si], [fl.d_SiO2])
    sample = lib.FilmStack([fl.n_air, n_tmdc, fl.n_fused_silica, fl.n_Si], [d_tmdc, fl.d_SiO2])
    r_blank = blank.RT(hw)[0]
    r_sample = sample.RT(hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    return contrast


def main(save):

    x20_image = root.reflection.x20_image

    x20rc = root.reflection.x20.copy()
    x20rc.contrast.clip(-.4, .4)
    x20rc.transform("energy", "ydist")
    x20rc.level(1, 0, -20)
    x20rc.smooth((20, 0))
    # translate y-axis to orient same junction as x100
    for d20x in [x20rc, x20_image]:
        ydist = d20x["ydist"]
        ydist[:] -=26.5
        ydist[:] *= -1

    ydist = x20_image["ydist"]
    ydist[:] += 2  # offset accounts for difference between image height with, w/o grating

    x100rc = root.reflection.x100
    x100rc.level(1, 0, -20)
    x100rc.transform("wm", "y")

    # --- plot ------------------------------------------------------------------------------------

    fig, gs = wt.artists.create_figure(width="double", cols=[1] * 3, wspace=1, hspace=0.75, nrows=2)
    axs = [plt.subplot(gs[0]), plt.subplot(gs[0, 1:])]
    

    divider = make_axes_locatable(axs[1])
    axCorr = divider.append_axes("right", 3.8, pad=0.2, sharey=axs[1])
    axs.append(axCorr)
    axs += [plt.subplot(gs[i]) for i in [3,4,5]]

    axs[0].pcolormesh(x20_image, cmap="gist_gray")
    axs[1].pcolormesh(x20rc, channel="contrast")
    axs[2].pcolormesh(x100rc, channel="contrast")

    axs[0].vlines([0], -10, 10, colors=["purple"], lw=2)
    axs[1].vlines([1e7/532 / 8065.5], -10, 10, colors=["green"], lw=2)
    axs[2].vlines([1e7/532 / 8065.5], -10, 10, colors=["green"], lw=2)

    for ax in axs[1:3]:
        ax.set_ylim(-10, 10)
        ax.set_xlim(1.7, 2.5)
        wt.artists.set_ax_spines(ax=ax, c="purple")
        ax.grid(True, **grid_kwargs)

    plt.setp(axs[2].get_yticklabels(), visible=False)
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.setp(axs[0].get_yticklabels(), visible=False)
    plt.setp(axs[4].get_yticklabels(), visible=False)

    [wt.artists.corner_text("abcdef"[i], ax=ax, distance=0.1) for i, ax in enumerate(axs)]

    wt.artists.corner_text("20x", ax=axs[1], corner="LR", distance=0.1, bbox=False)
    wt.artists.corner_text("100x", ax=axs[2], corner="LR", distance=0.1, bbox=False)

    # axs[0].indicate_inset_zoom(axs[1])
    axs[0].set_xlim(None, 50)
    [axs[0].tick_params(axis=i, which="both", length=0) for i in ["x", "y"]]

    axs[1].set_yticks([-10,-5, 0, 5, 10])

    # resonances vs position
    x100rc = x100rc.split("y", [-10, 10])[1]
    axs[2].plot(x100rc.ares.points, x100rc.y.points, color="k", ls="-", alpha=0.3)
    axs[2].plot(x100rc.bres.points, x100rc.y.points, color="k", ls="-", alpha=0.3)


    # rc slices and comparison with Fresnel predictions
    # wt.artists.corner_text(r"MoS$_2$", ax=axs[3], background_alpha=1, bbox=True, corner="LR")
    # wt.artists.corner_text(r"WS$_2$", ax=axs[4], background_alpha=1, bbox=True, corner="LR")

    # y_ws2 = root.reflection.x20.split("ydist", [15, 20])[1].contrast[:].mean(axis=1)
    y_ws2 = x20rc.split("ydist", [1.5, 2.5])[1].contrast[:].mean(axis=1)
    # y_mos2 = dx20.split("ydist", [31, 35])[1].contrast[:].mean(axis=1)
    y_mos2 = root.reflection.x20.split("ydist", [-41, -39])[1].contrast[:].mean(axis=1)
    x = root.reflection.x20.energy.points[:]
    axs[3].plot(x, y_mos2, lw=3, alpha=0.8, label="HS shell")
    axs[4].plot(x, y_ws2 * 1.5, lw=3, alpha=0.8, label="HS edge\n(x1.5)")

    # show ws2 control
    control = wt.open(here.parent / "data" / "zyz-554.wt5").reflection.refl
    control.transform("wm", "y")
    # out = wt.artists.interact2D(control, channel="subtr")
    # plt.show()
    # 1/0
    control = control.chop("wm", at={"y": [2, "um"]})[0]
    control.convert("eV")
    control.smooth(5)
    axs[4].plot(
        control, channel="subtr",
        lw=3, alpha=0.8, label="control", color="goldenrod")
    if False:
        control2 = wt.open(here.parent / "data" / "ws2 monolayers" / "na-assisted growth" / "root.wt5").spectrum
        control2 = control2.chop("wm", at={"y": [50, None]})[0]
        control2.convert("eV")
        control2.create_channel("rc_spot2", values=control2.spot2[:] / control2.signal[:] - 1, signed=True)
        control2.smooth(2)
        axs[4].plot(control2, channel="rc_spot2")

    # mos2 lineshape simulation
    # blank = lib.FilmStack([fl.n_air, fl.n_fused_silica, fl.n_Si], [fl.d_SiO2])
    # sample = lib.FilmStack([fl.n_air, nmos2, fl.n_fused_silica, fl.n_Si], [fl.d_mono, fl.d_SiO2])
    # r_blank = blank.RT(hw)[0]
    # r_sample = sample.RT(hw)[0]
    # contrast = (r_sample - r_blank) / r_blank
    # axs[4].plot(x, y_mos2, label=r"MoS$_2$", color="k", alpha=0.8)
    axs[3].plot(
        hw, sim_contrast(nmos2, fl.d_mono, hw),
        linewidth=2, linestyle=":", alpha=0.8, color="k", label=r"theory"
    )

    # ws2 lineshape simulation
    blank = lib.FilmStack([fl.n_air, fl.n_fused_silica, fl.n_Si], [fl.d_SiO2])
    sample = lib.FilmStack([fl.n_air, nws2, fl.n_fused_silica, fl.n_Si], [fl.d_mono, fl.d_SiO2])
    r_blank = blank.RT(hw)[0]
    r_sample = sample.RT(hw)[0]
    contrast = (r_sample - r_blank) / r_blank
    # axs[4].plot(x, y_mos2, label=r"MoS$_2$", color="k", alpha=0.8)
    axs[4].plot(
        hw, sim_contrast(nws2, fl.d_mono, hw),
        linewidth=2, linestyle=":", alpha=0.8, color="k", label=r"theory"
    )

    axs[3].grid(**grid_kwargs)
    axs[4].grid(**grid_kwargs)

    axs[3].set_ylim(axs[4].get_ylim())

    l1 = axs[3].legend(title=r"MoS$_2$", loc="lower right", fontsize=12)
    l2 = axs[4].legend(title=r"WS$_2$", loc="lower right", fontsize=12)

    axs[5].set_aspect(aspect="equal")
    axs[5].scatter(x100rc.ares.points, x100rc.bres.points, alpha=0.1, color="k")
    axs[5].scatter([1.963], [2.401], color="goldenrod")  # control resonance positions
    axs[5].grid(**grid_kwargs)
    wt.artists.set_ax_labels(
        ax=axs[5],
        xlabel=r"$\hbar\omega_A \ \left( \mathsf{eV} \right)$",
        ylabel=r"$\hbar\omega_B \ \left( \mathsf{eV} \right)$",        
    )
    axs[5].set_xticklabels([f"{_:0.2f}" for _ in axs[5].get_xticks()], rotation=-45)

 
    wt.artists.set_ax_labels(
        axs[1],
        ylabel=r"$y \ \left( \mu \mathsf{m}\right)$",
        xlabel=ev_label
    )
    wt.artists.set_ax_labels(axs[2], xlabel=ev_label)
    wt.artists.set_ax_labels(
        axs[3],
        ylabel=r"$\Delta R / R$",
        xlabel=ev_label
    )
    wt.artists.set_ax_labels(axs[4], xlabel=ev_label)

    for axi in [axs[3], axs[4]]:
        axi.set_xlim(1.6, 2.5)
        axi.hlines([0], *axi.get_xlim(), "k")

    if save:
        wt.artists.savefig(here / "reflection_contrast_v2.png")
    else:
        plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    main(save)
