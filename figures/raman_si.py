import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib


x0 = 12  # x-position (in um) to examine further
verbose = False

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "heterostructure.wt5"
root = wt.open(data_dir / p)
root.print_tree()
d = root.raman.proc

if False:
    dsags = root.PL.proc_PL
    wt.artists.interact2D(dsags, xaxis=2, yaxis=1)
    plt.show()


def run(save):
    """raman shift vs y position
    unlike main paper, also includes series spectral slices
    """

    d0 = d.chop("energy", "y", at={"x": [x0, "um"]})[0]
    # d0 = d0.split("energy", [440], units="wn")[0]
    d0.smooth((2, 0))
    d0.intensity.normalize()
    d0.create_channel("log_intensity", values=d0.intensity[:])
    d0.log_intensity.log10(floor=-2)
    

    fig, gs = wt.artists.create_figure(width="dissertation", cols=[1, "cbar", "cbar", "cbar", 0.6])
    ax0 = plt.subplot(gs[0])
    ax0.set_title(r"$x=" + f"{x0}" + r"\ \mathsf{\mu m}$")
    ax0.grid()
    ax0.set_ylim(-30, 30)
    ax0.pcolormesh(d0, channel="log_intensity", cmap="magma", vmin=d0.log_intensity.min())

    cax= plt.subplot(gs[1])
    ticks = np.linspace(-2, 0, 5)
    c = wt.artists.plot_colorbar(
        cax,
        cmap="magma",
        vlim=[d0.log_intensity.min(), d0.log_intensity.max()],
        ticks=ticks,
        label="log(intensity)"
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

    wt.artists.set_ax_labels(ax0, xlabel=r"Raman shift ($\mathsf{cm}^{-1}$)", ylabel=r"y-position $\left( \mathsf{\mu m} \right)$")
    wt.artists.set_ax_labels(ax1, xlabel=r"Raman shift ($\mathsf{cm}^{-1}$)")
    wt.artists.corner_text("a", ax=ax0)
    wt.artists.corner_text("b", ax=ax1)

    if save:
        p = f"raman_x_{x0}um.SI.png"
        p = here / p
        wt.artists.savefig(p)
    else: 
        plt.show()


def run2(save):
    """experimental figure for checking the mode splitting of MoS2
    """
    d_temp = d.chop("x", "y", at={"energy": [351, "wn"]}, verbose=verbose)[0]
    d_temp.intensity.normalize()
    separator = d_temp.intensity.points
    # MX2_area = separator > 0.2
    # WS2_area = separator > 0.36
    # MoS2_area = MX2_area & ~WS2_area

    # filter out WS2
    raman_lims = [150, 480]
    d0 = d.split("energy", raman_lims, verbose=verbose)[1]
    d0.create_variable("filter", values=separator[None, :, :])
    d0 = d0.split("filter", [0.1], verbose=verbose)[0]
    # baseline subtract from substrate
    # baseline varies across x, so correlating baseline value with x-position
    # also correlating with color
    d_temp = d0.chop("x", "y", at={"energy": [405, "wn"]}, verbose=verbose)[0]
    d_temp.intensity.normalize()
    d0.create_variable("filter2", values=d_temp.intensity.points[None, :, :])
    baseline = d0.split("filter2", [0.3], verbose=verbose)[0]
    z_baseline = np.nanmean(baseline.intensity.points, axis=2)[:, :, None]
    d0.intensity[:] -= z_baseline
    mos2 = d0.split("filter2", [0.3], verbose=verbose)[1]
    # further remove spurious substrate spectra
    mos2.create_variable("line1", values=(1.525 * mos2.x[:] - mos2.y[:] + 54.625), units="um")
    mos2.print_tree()
    mos2 = mos2.split("line1", [0])[1]
    mos2.smooth((2, 0, 0))
    # outa = wt.artists.interact2D(mos2)
    # plt.show()
    # mos2.level(0, 0, -5)  # axis=0 is energy

    # characterize the splitting
    modes = mos2.split("energy", [380.7, 387.7, 400.7, 413.7])
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

    if False:
        figa = plt.figure()
        ax = plt.subplot(111)
        spot1 = mos2.chop("energy", at={"x":[0, "um"], "y":[20, "um"]})[0]
        spot2 = mos2.chop("energy", at={"x":[-20, "um"], "y":[14, "um"]})[0]
        spot3 = mos2.chop("energy", at={"x":[0, "um"], "y":[40, "um"]})[0]
        ax.plot(spot1.energy.points, spot1.intensity[:])
        ax.plot(spot2.energy.points, spot2.intensity[:])
        ax.plot(spot3.energy.points, spot3.intensity[:])
        plt.show()

    mos2.create_channel("we", values=eprime.channels[-1].points[None, :, :])
    mos2.create_channel("wa", values=a1prime.channels[-1].points[None, :, :])

    ratio = eprime.channels[-2].points / a1prime.channels[-2].points
    splitting = a1prime.channels[-1].points - eprime.channels[-1].points
    mos2.create_channel("ratio", values=ratio[None, :, :])
    mos2.create_channel("splitting", values=splitting[None, :, :])
    mos2.ratio.clip(0.2, .5)
    mos2.splitting.clip(20.5, 25.5)
    mos2.we.clip(382.7, 384.7)
    mos2.wa.clip(405, 409)
    mos2.transform("x", "y")

    # figure 
    fig, gs = wt.artists.create_figure(
        width="dissertation", cols=[1,1,1], nrows=4, hspace=0.25,
        aspects=[[[0,0], 0.1], [[1,0], 1], [[2,0], 0.1], [[3,0], 1]]
    )

    # assemble
    # for color coding distributions
    splits = [22.8, 24.5]  # [22.8, 25]
    channel = (a1prime.channels[-1][:] - eprime.channels[-1][:]).flatten()
    colors = plt.cm.viridis([0, 0.5, 1])

    # plot
    ax0 = plt.subplot(gs[3,0])
    scatter_color = []
    for i in range(len(channel)):
        if channel[i] < splits[0]:
            scatter_color.append(colors[0])
        elif channel[i] > splits[1]:
            scatter_color.append(colors[2])
        else:
            scatter_color.append(colors[1])

    ax0.scatter(
        a1prime.channels[-1][:].flatten(),
        eprime.channels[-1][:].flatten(),
        s=10,
        color=scatter_color,
        linewidths=0,
        alpha=0.7
    )
    x = [403, 404.8, 405.8, 406.7]
    y = [384.2, 383.1, 382.7, 382.3]
    # ax0.scatter(x, y, marker="*", s=200, color="goldenrod")
    # for i, label in enumerate(["1L", "2L", "3L", "4L"]):
    #     ax0.annotate(label, (x[i], y[i]))

    # ax0.set_xlim(402.5, 409)

    wt.artists.set_ax_labels(
        ax0,
        xlabel=r"$\bar{\nu}_{A_1^\prime} \ \left( \mathsf{cm}^{-1} \right)$",
        ylabel=r"$\bar{\nu}_{E^\prime} \ \left( \mathsf{cm}^{-1} \right)$"
    )
    wt.artists.corner_text("d")
    ax0.set_yticks([382, 383, 384, 385])
    ax0.set_ylim(382, 385.5)
    
    ax1 = plt.subplot(gs[3,1])
    plt.yticks(visible=False)
    # ax1.vlines([18.8, 21.5, 23.3], ymin=0, ymax=50, color="goldenrod")
    bins = np.linspace(21, 26, 51)
    for bini, colori in zip(
        [
            [b for b in bins if b < splits[0]],
            [b for b in bins if (b > splits[0] and b < splits[1])],
            [b for b in bins if b > splits[1]]
        ],
        [colors[0], colors[1], colors[2]]
    ):
        bini.append(bini[-1] + 0.1)
        bini.append(bini[-1] + 0.1)
        ax1.hist(
            channel,
            bins=bini,
            color=colori,
            alpha=1
        )    
    ax1.set_ylim(0, 50)
    wt.artists.set_ax_labels(
        ax1,
        xlabel=r"$\bar{\nu}_{A_1^\prime}-\bar{\nu}_{E^\prime} \ \left( \mathsf{cm}^{-1} \right)$",
    )
    wt.artists.corner_text("e")

    ax2 = plt.subplot(gs[3, 2])
    ax2_inset = ax2.inset_axes([0.1, 0.6, 0.5, 0.35])

    mos2.create_variable("ae_splitting", values=mos2.splitting[:])
    rep_spectra = mos2.split("ae_splitting", splits)
    for color, spectrum in zip(colors, rep_spectra.values()):
        ax2.plot(spectrum.energy.points, np.nanmean(spectrum.intensity[:], axis=(1,2)), color=color)
        ax2_inset.plot(spectrum.energy.points, np.nanmean(spectrum.intensity[:], axis=(1,2)), color=color)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlim(*raman_lims)

    ax2_inset.set_xlim(375, 420)
    ax2_inset.set_xticks([385, 405])
    ax2_inset.set_yticks([])
    # ax2_inset.set_yticklabels([None for _ in ax2_inset.get_yticklabels()])

    for ax in [ax2, ax2_inset, ax1, ax0]:
        ax.set_facecolor([0.8] * 3)
        ax.grid(c="k", ls=":")

    wt.artists.corner_text("f")
    wt.artists.set_ax_labels(
        ax2,
        xlabel=r"$\mathsf{Raman \ shift} \ \left( \mathsf{cm}^{-1} \right)$",
        ylabel="average signal (a.u.)"
    )

    for col, channel, ticks, label in zip(
        [0, 1, 2],
        ["we", "wa", "splitting"],
        [np.linspace(382, 384, 5), np.linspace(400, 410, 11), np.linspace(20, 25, 6)],
        [
            r"$E_1^\prime \ \mathsf{mode} \ \left( \mathsf{cm}^{-1} \right)$",
            r"$A_1 \ \mathsf{mode} \ \left( \mathsf{cm}^{-1} \right)$",
            r"$\mathsf{splitting} \ \left( \mathsf{cm}^{-1} \right)$"
        ]
    ):

        vlim = [
            np.nanmin(mos2.channels[wt.kit.get_index(mos2.channel_names, channel)].points),
            np.nanmax(mos2.channels[wt.kit.get_index(mos2.channel_names, channel)].points)
        ]
        print(col, channel, vlim)
        axi = plt.subplot(gs[1, col])
        if col > 0:
            plt.yticks(visible=False)

        cmap = "viridis" if col !=0 else "viridis_r" 

        axi.pcolormesh(
            mos2,
            channel=channel,
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1]
        )

        caxi = plt.subplot(gs[0, col])
        wt.artists.plot_colorbar(
            caxi,
            cmap=cmap,
            orientation="horizontal",
            vlim=vlim,
            ticks=ticks,
            label=label
        )
        wt.artists.set_ax_labels(
            axi,
            xlabel=r"$x \ \left(\mathsf{\mu m} \right)$",
            ylabel=None if col > 0 else r"$y \ \left(\mathsf{\mu m} \right)$"
        )
        caxi.xaxis.set_label_position('top')
        caxi.xaxis.set_ticks_position('top')
        wt.artists.corner_text("abc"[col], ax=axi)
 
    if save:
        p = r"raman_mos2_mode_splitting.SI.png"
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
    # run(save)
    run2(save)
