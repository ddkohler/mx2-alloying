import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

def main(save):

    x20_image = root.reflection.x20_image

    x20rc = root.reflection.x20
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

    fig, gs = wt.artists.create_figure(width="double", cols=[1] * 3, wspace=1)
    axs = [plt.subplot(gs[0]), plt.subplot(gs[1:])]

    divider = make_axes_locatable(axs[1])
    axCorr = divider.append_axes("right", 3.8, pad=0.2, sharey=axs[1])
    axs.append(axCorr)

    axs[0].pcolormesh(x20_image, cmap="gist_gray")
    axs[1].pcolormesh(x20rc, channel="contrast")
    axs[2].pcolormesh(x100rc, channel="contrast")

    axs[0].vlines([0], -10, 10, colors=["purple"], lw=2)
    axs[1].vlines([1e7/532 / 8065.5], -10, 10, colors=["green"], lw=2)
    axs[2].vlines([1e7/532 / 8065.5], -10, 10, colors=["green"], lw=2)

    for ax in axs[1:]:
        ax.set_ylim(-10, 10)
        ax.set_xlim(1.7, 2.5)
        wt.artists.set_ax_spines(ax=ax, c="purple")
        ax.grid(True, ls=":", lw="1", c="k")

    plt.setp(axs[2].get_yticklabels(), visible=False)
    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.setp(axs[0].get_yticklabels(), visible=False)

    [wt.artists.corner_text("abc"[i], ax=ax) for i, ax in enumerate(axs)]

    wt.artists.corner_text("20x", ax=axs[1], corner="LR", distance=0.1, bbox=False)
    wt.artists.corner_text("100x", ax=axs[2], corner="LR", distance=0.1, bbox=False)

    # axs[0].indicate_inset_zoom(axs[1])
    axs[0].set_xlim(None, 50)
    [axs[0].tick_params(axis=i, which="both", length=0) for i in ["x", "y"]]

    [axs[i].set_xlabel(r"$\hbar \omega \ \left(\mathsf{eV}\right)$") for i in [1,2]]
    axs[1].set_ylabel(r"$y \ \left( \mu \mathsf{m}\right)$")
    axs[1].set_yticks([-10,-5, 0, 5, 10])

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
