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


angle_cmap = cm.get_cmap("hsv")
ref_cmap = cm.get_cmap("viridis")
colors = ref_cmap(np.linspace(0, 1, 6))

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")
screen = wt.open(here / "screen.wt5")

screen.print_tree()


def main(save=True):

    pl = root.pl.proc
    raman = root.raman.proc.split("energy", [500])[0]

    split_pl = []
    split_raman = []

    channels = ["junctiona", "junctionb", "mos2_core", "ws2"]
    for data in [pl, raman]:
        for ch in channels:
            data.create_variable(ch, values=screen[ch][:][None, :, :])

    deg = np.arctan2(pl.y[:], pl.x[:]) * 180 / np.pi
    degmod = deg / 360
    pl.create_variable("degmod", values=degmod)

    ws2_content = np.sqrt((screen["WS2_A1g"][:]-0.05)**2 + (screen["WS2_2LA"][:] - .02)**2)
    ws2_content *= 1.3
    pl.create_variable("ws2_content", values=ws2_content[None, :, :])
    pl_junction = pl.split("junctiona", [0.5])[1]
    pl_junction.transform("energy", "ws2_content")
    pl_junction.print_tree()
    fig, gs = wt.artists.create_figure(cols=[1,1,1])
    ax = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    m = (1.895 - 1.855) / 0.89
    b = 1.895 - m
    x = np.linspace(0.1, 1)
    ax.plot(x, m*x + b, "k:")

    for d in pl_junction.chop("energy").values():
        print(d.natural_name, d["ws2_content"][:])
        d.moment(0, channel=0, moment=1)
        y = d.intensity_energy_moment_1[:]
        x = d["ws2_content"][:]
        if not np.isnan(y):
            ax.scatter(
                x, y,
                color=angle_cmap(d.degmod[:] + 0.5)
            )
            above = (y - m * x - b) > 0
            print(above)
            ax1.scatter(d.x[:], d.y[:], c="b" if above else "r")
            # y = 1.855 + s * 0.2
            # m = 1.895 - 1.855 / .9 
            print(d.degmod[:])
            ax.text(x, y, f"{d.x[:]}, {d.y[:]}", fontsize=8)
            ax2.plot(d.energy.points, d.intensity[:], color=ref_cmap(d["ws2_content"][:]))
    plt.show()


if __name__ == "__main__":
    main(False)
