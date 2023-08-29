"""
check specific spectra for spaces that I have labelled as admixture regions
the conclusion seems to be that raman and pl confocal data do not perfectly line up; the fraction predicted by Raman
has a different relationship to peak, depending on whether the junction normal points upwards or downwards
=> consistent with a systematic offset of the data. 

"""

import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm


angle_cmap = cm.get_cmap("hsv")  # cyclic
ref_cmap = cm.get_cmap("viridis")
colors = ref_cmap(np.linspace(0, 1, 6))

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "heterostructure.wt5")
screen = wt.open(here.parent / "data" / "clusters.wt5")

# screen.print_tree()

# intervals to investigate
modes = dict(
    alloy_mode_1 = [394, 400],
    alloy_mode_2 = [375, 381],
    ws2_2la = [347, 353],
    ws2_a1g = [415, 418],
    mos2_e2g = [381.7, 385.7],
)


def main(save=True):

    fig, gs = wt.artists.create_figure(width="double", cols=[1,1], wspace=1., nrows=3)
    ax_map = plt.subplot(gs[0,0])
    ax_pl = plt.subplot(gs[0,1])

    corr_axes = [plt.subplot(gs[i]) for i in range(2,6)]

    ax1 = plt.subplot(gs[1])
    ax_map.set_xlabel("x (um)")
    ax_map.set_ylabel("y (um)")
    ax_pl = plt.subplot(gs[2])
    ax_pl.set_title("spectra")
    [axi.grid() for axi in fig.axes]

    for axi, (mode, interval) in zip(corr_axes, modes.items()):
        raman = root.raman.proc.split("energy", interval)[1]
        split_pl = []
        split_raman = []

        pl = root.pl.proc.copy()
        channels = ["junctiona", "junctionb", "mos2_core", "ws2"]
        for data in [pl, raman]:
            for ch in channels:
                data.create_variable(ch, values=screen[ch][:][None, :, :])
        raman.moment(2, channel=0, moment=0)  # integrate signal within split interval
        if "alloy" in pl.variable_names:
            pl.alloy[:] = raman.channels[-1][:]
        else:
            pl.create_variable("alloy", values=raman.channels[-1][:])
        # pick the cluster to investigate
        pl_junctiona = pl.split("junctiona", [0.5])[1]
        pl_junctionb = pl.split("junctionb", [0.5])[1]
        subsets = [pl_junctiona]

        # colormap options: based x and/or y values
        if False:
            # colormap looks at angular position
            map_color = lambda x, y: (np.arctan2(x,y) % (2*np.pi)) / (2 * np.pi)
            cmap = angle_cmap
        else:
            # map by y position only
            ylims = [-20, 20]
            map_color = lambda x, y: (y - ylims[0]) / (ylims[1] - ylims[0])
            cmap = ref_cmap


        ax.set_xlabel(f"Raman Signature ({mode})")
        ax.set_ylabel(f"PL Color Moment (eV)")

        for subset in subsets:
            for d in subset.chop("energy").values():
                if not np.isnan(d.intensity_energy_moment_1[:]):
                    color = cmap(map_color(d.x[:], d.y[:]))
                    y = d.intensity_energy_moment_1[0]
                    x = d["alloy"][:]
                    ax.scatter(
                        x, y, alpha=0.5, c=color
                    )
                    ax1.scatter(d.x[:], d.y[:], c=color)
                    # ax.text(x, y, f"{d.x[:]}, {d.y[:]}", fontsize=8)
                    ax2.plot(d.energy.points, d.intensity[:], c=color)
        if save:
            wt.artists.savefig(here / f"{mode} correlation.png")
        else:
            plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    main(save)
