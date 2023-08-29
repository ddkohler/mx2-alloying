import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib


verbose = False

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "heterostructure.wt5")
screen = wt.open(here.parent / "data" / "clusters.wt5")
plt.style.use(here / "figures.mplstyle")

# determines the ranges considered for analyzing E, A modes
line_splits = [380.7, 387.7, 400.7, 413.7]
colors = plt.cm.viridis([0.25, 0.6, 1])
# window for viewing raman shifts
raman_lims = [150, 480]
# bins for splitting energy (for color coding)
splits = [22.8, 24.5]  # [22.8, 25]
facecolor = [0.95] * 3
scatter_kwargs = dict(alpha=0.7, s=15, edgecolors=[0.5] * 3)

from matplotlib.colors import from_levels_and_colors
group_color, group = from_levels_and_colors(splits, colors, extend="both")
group_color = group_color.with_extremes(bad=facecolor)

d = root.raman.proc


def main(save):
    """experimental figure for checking the mode splitting of MoS2
    """
    mos2_screen = screen["mos2_edge"][:] + screen["mos2_core"][:]
    substrate_screen = screen["substrate"][:]

    d0 = d.split("energy", raman_lims, verbose=verbose)[1]
    d0.create_variable("mos2", values=mos2_screen[None, :, :])
    d0.create_variable("substrate", values=substrate_screen[None, :, :])

    # baseline subtract using substrate
    # baseline varies across x, so correlating baseline value with x-position
    # also correlating with color
    baseline = d0.split("substrate", [0.5], verbose=verbose)[1]
    z_baseline = np.nanmean(baseline.intensity.points, axis=2)[:, :, None]
    d0.intensity[:] -= z_baseline

    # isolate mos2
    mos2 = d0.split("mos2", [0.5], verbose=verbose)[1]
    # further remove spurious substrate spectra from outside structure
    mos2.create_variable("line1", values=(1.525 * mos2.x[:] - mos2.y[:] + 54.625), units="um")
    mos2 = mos2.split("line1", [0])[1]
    mos2.smooth((2, 0, 0))

    # characterize the splitting
    modes = mos2.split("energy", line_splits)
    for i, di in enumerate(modes.values()):
        if i in [0, 2, 4]:
            continue
        di.moment(channel=1, axis="energy", moment=0)
        if True:  # measure peak by averaging only large values position
            idxs = np.argmax(di.channels[1][:], axis=0)
            di.leveled[:] /= np.nanmax(di.leveled[:], axis=0)
            di.leveled.clip(min=0.5) # try to remove skew of peak
        di.moment(channel=1, axis="energy", moment=1)
    eprime = modes[1]
    a1prime = modes[3]

    mos2.create_channel("we", values=eprime.channels[-1].points[None, :, :])
    mos2.create_channel("wa", values=a1prime.channels[-1].points[None, :, :])

    ratio = eprime.channels[-2].points / a1prime.channels[-2].points
    splitting = a1prime.channels[-1].points - eprime.channels[-1].points
    mos2.create_channel("ratio", values=ratio[None, :, :])
    mos2.create_channel("splitting", values=splitting[None, :, :])
    mos2.ratio.clip(0.2, .5)
    mos2.transform("x", "y")

    # figure 
    fig, gs = wt.artists.create_figure(
        width="dissertation", cols=[1,1,1], nrows=4, hspace=0.25,
        aspects=[[[0,0], 0.1], [[1,0], 1], [[2,0], 0.1], [[3,0], 1]]
    )

    # correlation plot of A, E mode frequencies
    ax0 = plt.subplot(gs[3,0])

    ax0.scatter(
        a1prime.channels[-1][:].flatten(),
        eprime.channels[-1][:].flatten(),
        color = [i for i in map(group_color, group(splitting.flatten()))],
        **scatter_kwargs
    )

    ax0.set_xlabel(r"$\bar{\nu}_{A_1} \ \left( \mathsf{cm}^{-1} \right)$")
    ax0.set_ylabel(r"$\bar{\nu}_{E^\prime} \ \left( \mathsf{cm}^{-1} \right)$")

    wt.artists.corner_text("d")
    ax0.set_yticks([382, 383, 384, 385])
    ax0.set_ylim(382.5, 385.5)
    ax0.set_xlim(405, 409.2)


    sub_gs = gs[3,1].subgridspec(1, 4)
    ax1 = fig.add_subplot(sub_gs[0,1:])
    wt.artists.corner_text("e")

    temp = root.pl.proc
    temp.create_variable("mos2", values=mos2_screen[None, :, :])
    pl = temp.split("mos2", [0.5])[1]
    x = mos2.splitting[0].flatten()
    y = pl.intensity_energy_moment_1[0].flatten()

    ax1.scatter(
        x.flatten(),
        y.flatten(),
        color=[color for color in map(group_color, group(x))],
        **scatter_kwargs
    )
    ax1.set_ylabel(r"$\langle \hbar\omega \rangle_\mathsf{PL} \ (\mathsf{eV})$")
    ax1.set_xlabel(r"$A_1^\prime - E^\prime \ \mathsf{splitting} \ (\mathsf{cm}^{-1}$)")

    plt.ylim(1.82, 1.85)
    ax1.grid(True)

    ax2 = plt.subplot(gs[3, 2])
    ax2_inset = ax2.inset_axes([0.1, 0.6, 0.5, 0.35])

    mos2.create_variable("ae_splitting", values=mos2.splitting[:])
    rep_spectra = mos2.split("ae_splitting", splits)
    for color, spectrum in zip(colors, rep_spectra.values()):
        ax2.plot(spectrum.energy.points, np.nanmean(spectrum.leveled[:] + 3, axis=(1,2)), color=color)
        ax2_inset.plot(spectrum.energy.points, np.nanmean(spectrum.leveled[:], axis=(1,2)), color=color)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlim(*raman_lims)

    ax2_inset.set_xlim(375, 420)
    ax2_inset.set_xticks([385, 405])
    ax2_inset.set_yticks([])

    wt.artists.corner_text("f")
    ax2.set_xlabel(r"$\mathsf{Raman \ shift} \ \left( \mathsf{cm}^{-1} \right)$")
    ax2.set_ylabel("average signal (a.u.)")

    for col, channel, ticks, label, cmap, vlim in zip(
        [0, 1, 2],
        ["we", "wa", "splitting"],
        [np.linspace(382.5, 384.5, 5), np.linspace(400, 410, 11), np.linspace(20, 25, 6)],
        [
            r"$E_1^\prime \ \mathsf{mode} \ \left( \mathsf{cm}^{-1} \right)$",
            r"$A_1 \ \mathsf{mode} \ \left( \mathsf{cm}^{-1} \right)$",
            r"$A_1 - E_1^\prime \ \mathsf{splitting} \ \left( \mathsf{cm}^{-1} \right)$"
        ],
        [plt.cm.viridis_r, plt.cm.viridis, plt.cm.viridis],
        [
            [382.7, 384.7], [405, 409], [20.5, 25.5]
        ],
    ):
        if col == 2:  # mode splitting map is now a unique set of instructions
            continue

        axi = plt.subplot(gs[1, col])
        axi.set_yticks([-20, 0, 20, 40])
        if col > 0:
            plt.yticks(visible=False)

        cmap = cmap.with_extremes(over=cmap(1.), under=cmap(0.), bad=facecolor)
        kwargs = dict(cmap=cmap, vmin=vlim[0], vmax=vlim[1])

        axi.pcolormesh(mos2, channel=channel, **kwargs)

        caxi = plt.subplot(gs[0, col])
        wt.artists.plot_colorbar(
            caxi,
            cmap=cmap,
            orientation="horizontal",
            ticks=ticks,
            label=label,
            vlim=vlim,
            tick_fontsize=12,
            label_fontsize=14,
        )
        axi.set_xlabel(r"$x \ \left(\mathsf{\mu m} \right)$")
        axi.set_ylabel(None if col > 0 else r"$y \ \left(\mathsf{\mu m} \right)$")
        caxi.xaxis.set_label_position('top')
        caxi.xaxis.set_ticks_position('top')
        wt.artists.corner_text("abc"[col], ax=axi)

    # mode splitting spatial map
    ax_groups = plt.subplot(gs[1, 2])
    ax_groups.set_yticks([-20, 0, 20, 40])
    ax_groups.set_xlabel(r"$x \ \left(\mathsf{\mu m} \right)$")
    plt.yticks(visible=False)

    cmap, norm = group_color, group
    kwargs = dict(cmap=cmap, norm=norm)
    ax_groups.pcolormesh(mos2, channel="splitting", **kwargs)
    wt.artists.corner_text("c", ax=ax_groups)

    cax_groups = plt.subplot(gs[0, 2])
    from matplotlib import colorbar
    cb = colorbar.ColorbarBase(
        cax_groups,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
        label=r"$A_1 - E_1^\prime \ \mathsf{splitting} \ \left( \mathsf{cm}^{-1} \right)$",
        extendfrac=0.67,
        extendrect=True,
        ticklocation="top",
    )
    cb.ax.tick_params(labelsize=12)
    cb.set_label(label, fontsize=14)

    ax0a = ax_groups.inset_axes([0.7, 0.7, 0.4, 0.3])
    ax0a.set_yticks([])
    bins = np.linspace(21, 26, 51)
    for bini, colori in zip(
        [
            [b for b in bins if b < splits[0]],
            [b for b in bins if (b > splits[0] and b < splits[1])],
            [b for b in bins if b > splits[1]]
        ],
        colors
    ):
        bini.append(bini[-1] + 0.1)
        bini.append(bini[-1] + 0.1)
        ax0a.hist(
            splitting.flatten(),
            bins=bini,
            color=colori,
            alpha=1
        )
    ax0a.set_ylim(0, 45)
    ax0a.set_xticks([22, 24, 26])
    ax0a.grid(True, axis="x")
    # ax0a.set_title(r"$\Delta\bar{\nu}_{A_1^\prime E^\prime} \ \left( \mathsf{cm}^{-1} \right)$", fontsize=14)

    for ax in [ax2, ax2_inset, ax1, ax0, ax0a]:
        ax.set_facecolor(facecolor)
        ax.grid(c="k", ls=":")

    if save:
        p = r"raman_mos2.png"
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
