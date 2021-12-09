import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm


ref_cmap = cm.get_cmap("magma")
colors = ref_cmap(np.linspace(0, 1, 6))

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")
screen = wt.open(here / "screen.wt5")

screen.print_tree()

def main(save=True):

    pl = root.pl.proc
    raman = root.raman.proc

    split_pl = []
    split_raman = []

    channels = ["junctiona", "junctionb", "mos2_core", "ws2"]
    for data in [pl, raman]:
        for ch in channels:
            data.create_variable(ch, values=screen[ch][:][None, :, :])

    fig, gs = wt.artists.create_figure(width="single", cols=[1,1], default_aspect=2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    plt.yticks(visible=False)
    ax0.set_facecolor("gray")
    ax1.set_facecolor("gray")
    ax0.grid(b=True)
    ax1.grid(b=True)

    y0 = np.nanmean(pl.split("junctiona", [0.5])[1].intensity[:], axis=(1,2))
    y1 = np.nanmean(pl.split("junctionb", [0.5])[1].intensity[:], axis=(1,2))
    y2 = np.nanmean(pl.split("mos2_core", [0.5])[1].intensity[:], axis=(1,2))
    y3 = np.nanmean(pl.split("ws2", [0.5])[1].intensity[:], axis=(1,2))
    y4 = 0.4 * y2 + 0.4 * y3

    for y, c in zip([y0, y1, y2, y3], [colors[3], colors[4], colors[2], colors[5]]):
        ax0.plot(pl.energy.points, y, lw=2, alpha=0.7, c=c)
    ax0.plot(pl.energy.points, y4, lw=1, color="darkred", ls=":")

    y0 = np.nanmean(raman.split("junctiona", [0.5])[1].intensity[:], axis=(1,2))
    y1 = np.nanmean(raman.split("junctionb", [0.5])[1].intensity[:], axis=(1,2))
    y2 = np.nanmean(raman.split("mos2_core", [0.5])[1].intensity[:], axis=(1,2))
    y3 = np.nanmean(raman.split("ws2", [0.5])[1].intensity[:], axis=(1,2))
    y4 = 0.4 * y2 + 0.4 * y3

    for y, c in zip([y0, y1, y2, y3], [colors[3], colors[4], colors[2], colors[5]]):
        ax1.plot(raman.energy.points, y, lw=2, alpha=0.8, c=c)
    ax1.plot(raman.energy.points, y4, lw=1, color="darkred", ls=":")

    ax0.set_ylabel(r"$\mathsf{Intensity \ (a.u.)}$")
    ax1.set_ylim(-10, None)
    ax1.set_xlabel(r"$\mathsf{Raman \ shift} \ \left(\mathsf{cm}^{-1}\right)$")
    ax1.set_xlim(250, 500)
    ax0.set_xlabel(r"$\hbar \omega \ \left(\mathsf{eV}\right)$")

    if save:
        wt.artists.savefig(here / "junction_vs_admixture.png", fig=fig)
    else:
        plt.show()


from sys import argv
if len(argv) > 1:
    save = argv[1] != "0"
else:
    save = True
main(save)
