import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm


if True:  # pretty colors
    ref_cmap = cm.get_cmap("turbo_r")
    colors = [ref_cmap(i/5) for i in range(5)]
else:  # darker colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[3], colors[1], colors[2], colors[0], colors[4]]

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "heterostructure.wt5")
screen = wt.open(here.parent / "data" / "clusters.wt5")
logscale = False  # plot spectrum on logscale?

plt.style.use(here / "figures.mplstyle")


def plot_screen(ax, arr, color):
    cmap = ListedColormap([[1,1,1,0], color])
    ax.pcolormesh(screen.x.points, screen.y.points, arr.T, cmap=cmap)


def main(save):
    """display the grouping analysis (pub quality figure)
    """
    # alternative framing
    legend_kwargs={}
    fig, gs = wt.artists.create_figure(
        margin=[0.5, 0.3, 1.0, 0.3],
        width="double", cols=[1, 1, 1], nrows=2,
        wspace=0.8, hspace=0.85
    )
    ax_map = plt.subplot(gs[0,0])
    ax0 = plt.subplot(gs[0,1])
    ax1 = plt.subplot(gs[0,2])
    ax2 = plt.subplot(gs[1,1:])
    legend_kwargs = {"loc": "center left", "bbox_to_anchor": (0.06, 0.4), "ncol": 1}

    # ws2 ref spectrum
    ws2 = wt.open(here.parent / "data" / "ws2_control.wt5").raman.corner

    si_temp = ws2.split("x", [-30])[0].split("wm", [505, 530])[1] # isolate pure substrate
    si_temp.level(0, 0, 5) # remove baseline
    si_peak_sig = si_temp["signal"][:].mean(axis=1).max()

    # normalize by Si raman
    if True:  # remove baseline 
        ws2.level(0, 0, 5)
    ws2.signal[:] /= si_peak_sig
    ws2 = ws2.split("x", [-10, 10])[1]
    ws2.moment(1, 0, moment=0)
    ws2.channels[-1][:] /= ws2.shape[1]
    ws2.transform("wm")
    ax2.plot(ws2, channel=-1, color="k", lw=2, alpha=0.8)

    patches = []
    mean_spectra = []
    raman = root.raman.proc

    si_temp = root.raman.raw.split("y", [20])[1].split("x", [-20])[0].split("energy", [505, 530])[1] # isolate pure substrate
    si_temp.level(0, 0, 5) # remove baseline
    si_peak_sig = si_temp["intensity"][:].mean(axis=(1,2)).max()
    print(si_peak_sig, raman.intensity.max())

    raman.leveled[:] /= si_peak_sig

    # plot averaged raman spectrum for each region
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
        plot_screen(ax_map, screen[name][:], color)
        ax_map.set_facecolor([1] * 3)

        # scatter
        for ax, xchan, ychan in [[ax0, "MoS2_A1g", "MoS2_E2g"], [ax1, "WS2_A1g", "WS2_2LA"]]:
            mask = screen[name][:]
            x = screen[xchan][:][mask].flatten()
            y = screen[ychan][:][mask].flatten()
            alpha = 1 if name in ["junctiona", "junctionb"] else 0.5
            ax.scatter(x, y, s=10, alpha=alpha, color=color, edgecolor="none")

        # spectrum
        zone = screen[name][:]
        cmap = ListedColormap([[0,0,0,0], color])
        patches.append(mpatches.Patch(color=color, label=label))
        raman.create_variable(name=name, values=zone[None, :, :])
        split = raman.split(name, [0.5], verbose=False)[1]
        y = np.nanmean(split.leveled[:], axis=(1,2))
        ax2.plot(split.energy.points, y, color=color, lw=3, alpha=1)
        mean_spectra.append(y)

    legend = fig.legend(
        handles=patches,
        **legend_kwargs
    )

    ax2.annotate(
        r"pure WS$_2$",
        xy=(355, 1.9),
        fontsize=14,
        xytext=(365, 1.3),
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3"),
    )

    for ax in [ax0, ax1, ax2]:
        ax.grid()

    for ax in [ax0, ax1]:
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks(np.linspace(0,1,2))
        ax.set_yticks(np.linspace(0,1,2))

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    scalebar = AnchoredSizeBar(
        ax_map.transData,
        50, r'50 $\mathsf{\mu}$m',
        'lower left', 
        pad=0.3,
        color='black',
        frameon=False,
        size_vertical=1,
        fill_bar=True
    )
    ax_map.add_artist(scalebar)

    ax2.set_xlim(150, 450)
    if logscale:
        ax2.set_ylim(0.03, 2.1)
        ax2.set_yscale("log")
    else:
        ax2.set_ylim(-0.1, 2.1)
        ax2.set_yticks(np.linspace(0,2,3))

    ax0.set_xlabel(r"$\mathsf{MoS_2 \ A_1^\prime \ intensity \ (norm.)}$")
    ax0.set_ylabel(r"$\mathsf{MoS_2 \ E^{\prime} \ intensity \ (norm.)}$")
    ax1.set_xlabel(r"$\mathsf{WS_2 \ A_1^\prime \ intensity \ (norm.)}$")
    ax1.set_ylabel(r"$\mathsf{WS_2 \ 2LA \ intensity \ (norm.)}$")
    ax2.set_xlabel(r"$\mathsf{Raman \ shift} \ \left(\mathsf{cm}^{-1}\right) $")
    ax2.set_ylabel(r"$\mathsf{Intensity \ (a.u.)}$")

    for i, ax in enumerate(fig.axes):
        wt.artists.corner_text("abcd"[i], ax=ax)

    if save:
        p = "separator.png"
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
    main(save)

