# table of contents art matter

import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.colors import ListedColormap

here = pathlib.Path(__file__).resolve().parent


def plot_screen(ax, arr, color):
    cmap = ListedColormap([[1,1,1,0], color])
    ax.pcolormesh(screen.x.points, screen.y.points, arr.T, cmap=cmap)


hs = wt.open(here.parent / "data" / "heterostructure.wt5")
screen = wt.open(here.parent / "data" / "clusters.wt5")


def main():
    fig, gs = wt.artists.create_figure(cols=[1,1])
    ax0 = plt.subplot(gs[0])
    ax0.pcolormesh(hs.pl.proc, channel=1, cmap="rainbow_r", norm=Normalize(1.82, 1.94))
    ax0.set_facecolor("white")
    plt.xticks(visible=False)

    ax1 = plt.subplot(gs[1])
    ref_cmap = cm.get_cmap("turbo_r")
    colors = [ref_cmap(i/5) for i in range(5)]
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    for color, (label, name) in zip(
        colors,
        {
            # r"$\mathsf{substrate}$": "substrate",
            r"$\mathsf{MoS}_2 \ \mathsf{(edge)}$": "mos2_edge",
            r"$\mathsf{MoS}_2$": "mos2_core",
            r"$\mathsf{heterojunction}$": "junctiona",
            r"$\mathsf{WS}_2 \ \mathsf{(edge)}$": "junctionb",
            r"$\mathsf{WS}_2 \ \mathsf{(core)}$": "ws2",
        }.items()
    ):
        # map
        plot_screen(ax1, screen[name][:], color)
        ax1.set_facecolor([1] * 3)


    plt.show()


if __name__ == "__main__":
    main()