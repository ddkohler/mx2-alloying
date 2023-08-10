import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib


verbose = False

here = pathlib.Path(__file__).resolve().parent
name = pathlib.Path(__file__).resolve().name
root = wt.open(here.parent / "data" / "heterostructure.wt5")
# screen = wt.open(here.parent / "data" / "clusters.wt5")
plt.style.use(here / "figures.mplstyle")
colors = [plt.cm.tab10(i) for i in range(6)]

cmap = wt.artists.colormaps["default"]
cmap = cmap.with_extremes(under=[1]*3)  #, bad=[0]*3)



# coordinates
xy = {
    -12: [-24, -12, 0, 28],
    20: [-6, -2, 0, 2, 6, 14],
    22: [-16, 2]
}


def main(save):

    d = root.raman.proc.split("energy", [500])[0]

    fig, gs = wt.artists.create_figure(width="double", cols=[1,1,1,"cbar"], nrows=2)

    for col, (x, ys) in enumerate(xy.items()):
        axi0 = plt.subplot(gs[0,col])
        plt.xticks(visible=False)
        if col > 0:
            plt.yticks(visible=False)
        axi1 = plt.subplot(gs[1,col])
        if col > 0:
            plt.yticks(visible=False)
        else:
            axi0.set_ylabel(r"$\mathsf{y \ (\mu m)}$")
            axi1.set_ylabel(r"$\mathsf{Intensity \ (a.u.)}$")

        axi1.set_xlabel(r"$\mathsf{Raman \ Shift \ (cm^{-1})}$")

        di = d.at(x=[x, "um"])
        di.smooth((2,0))
        di.transform("energy", "y")
        axi0.pcolormesh(di, channel="intensity", cmap=cmap)

        for i, y in enumerate(ys):
            axi0.hlines([y], di.energy.min(), di.energy.max(), color=colors[i], alpha=0.7)
            dy = di.at(y=[y, "um"])
            axi1.plot(dy.energy[:], dy.intensity[:], label=r"$\mathsf{y=" + f"{y}" + r"\ \mu m}$", c=colors[i], alpha=0.7)

        axi0.set_title(r"$\mathsf{x=" + f"{x}" + r"\ \mu m}$")
        axi1.set_xlim(d.energy.min(), d.energy.max())
        axi1.legend(fontsize=14)
        axi0.grid(True)
        axi1.grid(True)
        

    cax = plt.subplot(gs[0,3])
    wt.artists.plot_colorbar(cax=cax, cmap=cmap, label="Intensity (a.u.)", extend="min")

    if save:
        p = here / f"{name}.png"
        wt.artists.savefig(p, facecolor="white")
    else:
        plt.show()


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        save = argv[1] != "0"
    else:
        save = True
    main(save)

