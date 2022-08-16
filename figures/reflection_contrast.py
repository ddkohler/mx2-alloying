import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
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


def run(save):

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

    fig, gs = wt.artists.create_figure(
        width="double", cols=[1] * 3, wspace=1, hspace=0.15, nrows=4,
        aspects=[
            [[0,0], 0.1],
            [[2,0], 0.15]
        ],
    )
    axs = [plt.subplot(gs[1,0]), plt.subplot(gs[1, 1:])]
    cax= plt.subplot(gs[0,1:])
    wt.artists.plot_colorbar(
        cax,
        ticks=np.linspace(-1,1,11),
        cmap="signed",
        orientation="horizontal",
        ticklocation="top",
        label=r"$\Delta R / R \ (\mathsf{norm})$"
    )

    divider = make_axes_locatable(axs[1])
    axCorr = divider.append_axes("right", 3.8, pad=0.2, sharey=axs[1])
    axs.append(axCorr)
    axs += [plt.subplot(gs[3, i]) for i in [0,1,2]]

    axs[0].pcolormesh(x20_image, cmap="gist_gray", vmin=1e5)
    axs[1].pcolormesh(x20rc, channel="contrast")
    axs[2].pcolormesh(x100rc, channel="contrast")

    axs[0].vlines([0], -10, 10, colors=["purple"], lw=2)
    axs[1].vlines([1e7/532 / 8065.5], -10, 10, colors=["green"], lw=2)
    axs[2].vlines([1e7/532 / 8065.5], -10, 10, colors=["green"], lw=2)

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    scalebar = AnchoredSizeBar(
        axs[0].transData,
        50, r'50 $\mathsf{\mu}$m', 'lower left', 
        pad=0.3,
        color='black',
        frameon=False,
        size_vertical=2,
        fill_bar=True
    )
    axs[0].add_artist(scalebar)

    for ax in axs[1:3]:
        ax.set_ylim(-10, 10)
        ax.set_xlim(1.7, 2.5)
        wt.artists.set_ax_spines(ax=ax, c="purple")
        ax.grid(True, **grid_kwargs)

    plt.setp(axs[2].get_yticklabels(), visible=False)
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.setp(axs[0].get_yticklabels(), visible=False)
    plt.setp(axs[4].get_yticklabels(), visible=False)

    # subplot labels, annotation
    [wt.artists.corner_text("abcdef"[i], ax=ax, distance=0.1) for i, ax in enumerate(axs[:-1])]
    axs[-1].text(-0.75, 0.95, "f", transform=axs[-1].transAxes, fontsize=18,
        bbox=dict(boxstyle="square", facecolor="white", edgecolor=matplotlib.rcParams["legend.edgecolor"])
    )
    axs[1].text(
        0.5, 0.95, "NA 0.46", va="top", ha="center", transform=axs[1].transAxes, fontsize=18, 
        path_effects=[pe.withStroke(linewidth=3, foreground="w")],
    )
    axs[2].text(
        0.5, 0.95, "NA 0.96", va="top", ha="center", transform=axs[2].transAxes, fontsize=18, 
        path_effects=[pe.withStroke(linewidth=3, foreground="w")],
    )

    # axs[0].indicate_inset_zoom(axs[1])
    axs[0].set_xlim(None, 50)
    [axs[0].tick_params(axis=i, which="both", length=0) for i in ["x", "y"]]

    axs[1].set_yticks([-10,-5, 0, 5, 10])

    # resonances vs position
    x100rc = x100rc.split("y", [-10, 10])[1]
    axs[2].plot(x100rc.ares.points, x100rc.y.points, color="k", ls="-", alpha=0.3)
    axs[2].plot(x100rc.bres.points, x100rc.y.points, color="k", ls="-", alpha=0.3)

    # rc slices and comparison with Fresnel predictions
    y_ws2 = x20rc.split("ydist", [1.5, 2.5])[1].contrast[:].mean(axis=1)
    y_mos2 = root.reflection.x20.split("ydist", [-41, -39])[1].contrast[:].mean(axis=1)
    x = root.reflection.x20.energy.points[:]
    axs[3].plot(x, y_mos2, lw=3, alpha=0.8, label="HS shell")
    axs[4].plot(x, y_ws2 * 1.5, lw=3, alpha=0.8, label="HS edge\n(x1.5)")

    # show ws2 control
    control = wt.open(here.parent / "data" / "zyz-554.wt5").reflection.refl
    control.transform("wm", "y")
    control = control.chop("wm", at={"y": [2, "um"]})[0]
    control.convert("eV")
    control.smooth(5)
    axs[4].plot(
        control, channel="subtr",
        lw=3, alpha=0.8, label="control", color="goldenrod")

    # mos2 lineshape simulation
    axs[3].plot(
        hw, sim_contrast(nmos2, fl.d_mono, hw),
        linewidth=2, linestyle=":", alpha=0.8, color="k", label=r"theory"
    )

    # ws2 lineshape simulation
    axs[4].plot(
        hw, sim_contrast(nws2, fl.d_mono, hw),
        linewidth=2, linestyle=":", alpha=0.8, color="k", label=r"theory"
    )

    axs[3].grid(**grid_kwargs)
    axs[4].grid(**grid_kwargs)

    axs[3].set_ylim(axs[4].get_ylim())

    l1 = axs[3].legend(title=r"MoS$_2$", loc="lower right", fontsize=12)
    l2 = axs[4].legend(title=r"WS$_2$", loc="lower right", fontsize=12)

    axs[5].scatter(x100rc.ares.points, x100rc.bres.points, alpha=0.1, color="k", linewidths=0)
    axs[5].scatter([1.963], [2.401], color="goldenrod", s=200, marker="D", alpha=0.7, edgecolors="k", linewidths=0)  # control resonance positions
    axs[5].grid(**grid_kwargs)
    wt.artists.set_ax_labels(
        ax=axs[5],
        xlabel=r"$\hbar\omega_A \ \left( \mathsf{eV} \right)$",
        ylabel=r"$\hbar\omega_B \ \left( \mathsf{eV} \right)$",        
    )
    axs[5].set_xticklabels([f"{_:0.2f}" for _ in axs[5].get_xticks()], rotation=-45)
    axs[5].set_xlim(1.8, 2.0)
    axs[5].set_ylim(2.0, 2.45)
    axs[5].set_aspect("equal")
 
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
    run(save)
