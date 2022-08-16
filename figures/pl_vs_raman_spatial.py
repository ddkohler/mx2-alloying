import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt


def run(save):
    here = pathlib.Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    p = "data.wt5"
    root = wt.open(data_dir / p)
    pl = root.pl.proc.copy()

    pl.smooth((5, 0, 0))
    pl_color = pl.intensity_energy_moment_1[0]
    pl_color[pl_color < 1.81] = 1.81

    raman = root.raman.proc
    raman1 = raman.chop("x", "y", at={"energy": [416, "wn"]})[0]
    raman2 = raman.chop("x", "y", at={"energy": [350, "wn"]})[0]

    fig, gs = wt.artists.create_figure(width="dissertation", cols=[1, 1, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)
    plt.yticks(visible=False)
    ax2 = plt.subplot(gs[2], sharey=ax0)
    plt.yticks(visible=False)

    ax1.set_title(r"Raman, WS$_2$ 2LA(M)")
    ax0.set_title(r"Raman, WS$_2$ A$_1$($\mathsf{\Gamma}$)")
    ax2.set_title(r"PL $\langle \hbar \omega \rangle$")

    ax0.pcolormesh(raman1, channel="leveled", cmap="magma")
    ax1.pcolormesh(raman2, channel="leveled", cmap="magma")
    ax2.pcolormesh(pl.axes[0].points, pl.axes[1].points, pl_color.T, cmap="rainbow_r")

    for axi in [ax0, ax1, ax2]:
        axi.grid()
        axi.set_ylim(-25, 25)
        axi.set_xlim(-25, 25)
    wt.artists.set_ax_labels(ax0, xlabel=r"$x \ (\mu\mathsf{m})$", ylabel=r"$y \ (\mu\mathsf{m})$")
    wt.artists.set_ax_labels(ax1, xlabel=r"$x \ (\mu\mathsf{m})$")
    wt.artists.set_ax_labels(ax2, xlabel=r"$x \ (\mu\mathsf{m})$")

    # save
    if save:
        p = "raman_pl_comparison.png"
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

