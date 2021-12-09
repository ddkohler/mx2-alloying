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


raman = root.raman.proc.split("energy", [500])[0]
pl = root.pl.proc

for d in [raman, pl]:
    deg = np.arctan2(d.y[:], d.x[:]) * 180 / np.pi
    degmod = deg % 120
    degmod -= 9
    d.create_variable("degmod", values=degmod)
    channels = ["junctiona", "junctionb", "mos2_core", "ws2"]
    for ch in channels:
        d.create_variable(ch, values=screen[ch][:][None, :, :])

    junction = d.split("junctiona", [0.5])[1]

    alpha = junction.split("degmod", [0, 40])[1]
    beta = junction.split("degmod", [40])[1]

    fig, gs = wt.artists.create_figure()
    ax = plt.subplot(gs[0])
    for data in [alpha, beta]:
        y = np.nanmean(data.intensity[:], axis=(1,2))
        # y /= y.sum()
        ax.plot(data.energy.points, y)

plt.show()
