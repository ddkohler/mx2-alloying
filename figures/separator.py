import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import cm

ref_cmap = cm.get_cmap("turbo_r")
colors = [ref_cmap(i/5) for i in range(5)]

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")
screen = wt.open(here.parent / "fitting" / "screen.wt5")

plt.style.use(here / "figures.mplstyle")

# ws2.print_tree()
# out = wt.artists.interact2D(ws2.raman.face)
# plt.show()


# an exploration:
# does PL emission frequency correlate with 2LA brightness?
# if False:
#     fig, gs = wt.artists.create_figure(cols=[1,1])

#     ax0 = plt.subplot(gs[0])
#     ax0.pcolormesh(root.pl.proc, channel="intensity_energy_moment_1")

#     ax1 = plt.subplot(gs[1])
#     ax1.pcolormesh(screen, channel="WS2_2LA")

# if False:
#     fig, gs = wt.artists.create_figure()

#     ax0 = plt.subplot(gs[0])
#     for filt in [np.logical_not(screen.ws2[:]), np.logical_not(screen.junctionb[:])]:
#         ws2_2la = screen.WS2_2LA[:].copy()
#         ws2_2la[filt] *= np.nan

#         ax0.scatter(
#             root.pl.proc.intensity_energy_moment_1[:].flatten(),
#             ws2_2la.flatten(),
#             alpha=0.2
#         )
#     plt.show()


def plot_screen(ax, arr, color):
    cmap = ListedColormap([[0,0,0,0], color])
    ax.pcolormesh(screen.x.points, screen.y.points, arr.T, cmap=cmap)


def main(save):
    """display the grouping analysis (pub quality figure)
    """
    # alternative framing
    if False:
        fig, gs = wt.artists.create_figure(
            width="double", cols=[1, 1, 1, 1], wspace=1
        )
        ax_map = plt.subplot(gs[0:1], aspect=1)
        ax0 = plt.subplot(gs[1])
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax0)
        ax1 = divider.append_axes("bottom", 3, pad=1, sharex=ax0)
        ax2 = plt.subplot(gs[2])
    else:
        fig, gs = wt.artists.create_figure(
            margin=[1.0, 0.2, 1.0, 0.3],
            width="double", cols=[1, 1, 1, 1], wspace=1, hspace=0.75
        )
        ax_map, ax0, ax1, ax2 = [plt.subplot(gs[i]) for i in range(4)]

    # ws2 ref spectrum
    ws2 = wt.open(here.parent / "data" / "zyz-554.wt5").raman.corner

    si_temp = ws2.split("x", [-30])[0].split("wm", [505, 530])[1] # isolate pure substrate
    si_temp.level(0, 0, 5) # remove baseline
    si_peak_sig = si_temp["signal"][:].mean(axis=1).max()

    # normalize by Si raman
    if True:  # remove baseline 
        ws2.level(0, 0, 5)
    ws2.signal[:] /= si_peak_sig
    # subtract Si Raman
    # ws2.level(0, 1, 3)
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
        # ax0.pcolormesh(screen.x.points, screen.y.points, zone.T, cmap=cmap)
        raman.create_variable(name=name, values=zone[None, :, :])
        split = raman.split(name, [0.5], verbose=False)[1]
        y = np.nanmean(split.leveled[:], axis=(1,2))
        ax2.plot(split.energy.points, y, color=color, lw=2, alpha=0.8)
        mean_spectra.append(y)

    legend = fig.legend(
        handles=patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.94),
        ncol=5
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
        ax.set_xticks(np.linspace(0,1,3))
        ax.set_yticks(np.linspace(0,1,3))

    ax_map.set_xticks([])
    ax_map.set_yticks([])
    # ax_map.set_xticklabels([None for _ in ax_map.get_xticklabels()])
    # ax_map.set_yticklabels([None for _ in ax_map.get_yticklabels()])

    ax2.set_xlim(250, 450)
    ax2.set_ylim(-0.1, 2.1)

    ax0.set_xlabel(r"$\mathsf{MoS_2 \ A_1^\prime \ intensity \ (norm.)}$")  # , fontsize=12)
    ax0.set_ylabel(r"$\mathsf{MoS_2 \ E^{\prime} \ intensity \ (norm.)}$")  # , fontsize=12)
    ax1.set_xlabel(r"$\mathsf{WS_2 \ A_1^\prime \ intensity \ (norm.)}$")  # , fontsize=12)
    ax1.set_ylabel(r"$\mathsf{WS_2 \ 2LA \ intensity \ (norm.)}$")  # , fontsize=12)
    ax2.set_xlabel(r"$\mathsf{Raman \ shift} \ \left(\mathsf{cm}^{-1}\right) $")  # , fontsize=12)
    ax2.set_ylabel(r"$\mathsf{Intensity \ (a.u.)}$")  # , fontsize=12)

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

