import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib


x0 = -12  # x-position (in um) to examine further
verbose = False

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "heterostructure.wt5"
root = wt.open(data_dir / p)
root.print_tree()
d = root.raman.raw

cmap = plt.cm.magma.copy()
cmap = cmap.with_extremes(under=[0]*3, bad=[0]*3)

if False:
    dsags = root.PL.proc_PL
    wt.artists.interact2D(dsags, xaxis=2, yaxis=1)
    plt.show()


def run(save):
    """raman shift vs y position
    unlike main paper, also includes series spectral slices
    """

    d0 = d.chop("energy", "y", at={"x": [x0, "um"]})[0]
    d0.smooth((2, 0))
    d0.intensity.normalize()
    d0.create_channel("log_intensity", values=d0.intensity[:])
    d0.log_intensity.log10(floor=-2)
    

    fig, gs = wt.artists.create_figure(width="dissertation", cols=[1, "cbar", "cbar", "cbar", 0.6])
    ax0 = plt.subplot(gs[0])
    ax0.set_title(r"$x=" + f"{x0}" + r"\ \mathsf{\mu m}$")
    ax0.grid()
    ax0.set_ylim(-30, 30)
    ax0.pcolormesh(d0, channel="log_intensity", cmap=cmap, vmin=d0.log_intensity.min())

    cax= plt.subplot(gs[1])
    ticks = np.linspace(-2, 0, 5)
    c = wt.artists.plot_colorbar(
        cax,
        cmap=cmap,
        vlim=[d0.log_intensity.min(), d0.log_intensity.max()],
        ticks=ticks,
        label="log(intensity)",
        extend="min"
    )

    colors = plt.cm.viridis_r(np.linspace(0, 1, 10))

    ax1 = plt.subplot(gs[4])
    plt.yticks(visible=False)
    ax1.vlines([174, 350, 416], -1, 10, ls=":", color="k", linewidth=1)
    ax1.vlines([383, 406], -1, 10, ls="--", color="k", linewidth=1)
    for i, yi in enumerate(range(-28, -8, 2)):
        di  = d0.chop("energy", at={"y": [yi, "um"]})[0]
        # di.intensity.normalize()
        di.intensity[:] += i * 0.25
        ax1.plot(di, channel="intensity", color=colors[i])
        ax0.hlines([yi], 100, 600, color=colors[i])
    # ax1.grid()
    ax1.set_ylim(0, 3)
    ax1.set_xlim(100, 600)
    # ax1.set_yticks()

    wt.artists.set_ax_labels(ax0, xlabel=r"$\mathsf{Raman \ shift \ (cm^{-1})}$", ylabel=r"$\mathsf{y \ position \ \left( \mu m \right)}$")
    wt.artists.set_ax_labels(ax1, xlabel=r"$\mathsf{Raman \ shift \ (cm^{-1})}$")
    wt.artists.corner_text("a", ax=ax0)
    wt.artists.corner_text("b", ax=ax1)

    if save:
        p = f"raman_x_{x0}um.SI.png"
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
