import numpy as np
import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib


x0 = 12  # x-position (in um) to examine further
verbose = False

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "ZYZ543.wt5"
root = wt.open(data_dir / p)
d = root.raman.proc_raman

if False:
    dsadfj = root.PL.raw_PL
    dsadfj.level(0, 2, -2)
    dsadfj.convert("eV")
    outs = wt.artists.interact2D(dsadfj, xaxis="energy")
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
    d0.log_intensity.log10()

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
    d_temp = d.chop("x", "y", at={"energy": [350, "wn"]}, verbose=verbose)[0]
    d_temp.intensity.normalize()
    separator = d_temp.intensity.points
    # MX2_area = separator > 0.2
    # WS2_area = separator > 0.36
    # MoS2_area = MX2_area & ~WS2_area

    # filter out WS2
    d0 = d.split("energy", [370, 420], verbose=verbose)[1]
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
    # outa = wt.artists.interact2D(mos2, xaxis="energy")
    mos2.level(0, 0, -2)  # axis=0 is energy

    # characterize the splitting
    modes = mos2.split("energy", [380, 387, 400, 413])
    for di in modes.values():
        di.moment(channel=1, axis="energy", moment=1)
        # di.create_channel(
        #     "max", 
        #     values=di.energy.points[np.argmax(di.intensity.points, axis=0)]
        # )
        di.moment(channel=1, axis="energy", moment=0)
    eprime = modes[1]
    a1prime = modes[3]

    # figure 
    fig, gs = wt.artists.create_figure(
        width="dissertation", cols=[1,1,1], nrows=4, hspace=0.25,
        aspects=[[[0,0], 0.1], [[1,0], 1], [[2,0], 0.1], [[3,0], 1]]
    )

    # axes[5].plot(mos2.energy.points, np.nanmean(mos2.intensity.points, axis=(1,2)))

    # assemble
    # plot
    ax0 = plt.subplot(gs[3,0:2])
    ax0.scatter(
        a1prime.channels[-2][:].flatten(),
        eprime.channels[-2][:].flatten(),
        s=10, color="k", alpha=0.3)
    x = [403, 404.8, 405.8, 406.7]
    y = [384.2, 383.1, 382.7, 382.3]
    ax0.scatter(x, y, marker="*", s=100, color="goldenrod")
    for i, label in enumerate(["1L", "2L", "3L", "4L"]):
        ax0.annotate(label, (x[i], y[i]))

    ax0.set_ylim(382, 384.5)
    ax0.set_xlim(402.5, 408)
    wt.artists.set_ax_labels(
        ax0,
        xlabel=r"$\bar{\nu}_{A_{1g}} \ \left( \mathsf{cm}^{-1} \right)$",
        ylabel=r"$\bar{\nu}_{E_{2g}^\prime} \ \left( \mathsf{cm}^{-1} \right)$"
    )
    wt.artists.corner_text("d")
    ax0.grid()
    ax0.set_yticks([382, 383, 384])
    
    ax1 = plt.subplot(gs[3,2])
    ax1.vlines([18.8, 21.5, 23.3], ymin=0, ymax=50, color="goldenrod")
    ax1.hist(
        (a1prime.channels[-2][:] - eprime.channels[-2][:]).flatten(),
        bins=np.linspace(20, 25),
        color="k",
        alpha=0.5
    )
    ax1.set_ylim(0, 50)
    wt.artists.set_ax_labels(
        ax1,
        xlabel=r"$\bar{\nu}_{A_{1g}}-\bar{\nu}_{E_{2g}^\prime} \ \left( \mathsf{cm}^{-1} \right)$",
        ylabel="counts"
    )
    ax1.yaxis.set_label_position("right")
    wt.artists.corner_text("e")
    plt.grid()
    ax1.yaxis.tick_right()

    if True:  # spatial maps of peaks
        mos2.create_channel("we", values=eprime.channels[-2].points[None, :, :])
        mos2.create_channel("wa", values=a1prime.channels[-2].points[None, :, :])

        ratio = eprime.channels[-1].points / a1prime.channels[-1].points
        splitting = a1prime.channels[-2].points - eprime.channels[-2].points
        mos2.create_channel(
            "ratio",
            values=ratio[None, :, :]
        )
        mos2.create_channel(
            "splitting",
            values=splitting[None, :, :]
        )
        mos2.print_tree()
        mos2.ratio.clip(0.2, .5)
        mos2.splitting.clip(22, 25)
        mos2.we.clip(382.5, 384)
        mos2.wa.clip(405, 408)
        mos2.transform("x", "y")

        for col, channel, ticks, label in zip(
            [0, 1, 2],
            ["we", "wa", "splitting"],
            [np.linspace(382, 385, 7), np.linspace(400, 410, 11), np.linspace(20, 25, 6)],
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
            axi = plt.subplot(gs[1, col])
            if col > 0:
                plt.yticks(visible=False)
            axi.pcolormesh(
                mos2,
                channel=channel,
                cmap="viridis",
                vmin=vlim[0],
                vmax=vlim[1]
            )

            caxi = plt.subplot(gs[0, col])
            wt.artists.plot_colorbar(
                caxi,
                cmap="viridis",
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

    # out = wt.artists.interact2D(d1, channel=-2)
    # modes = d0.split("energy", [392])

    # print(z_baseline.shape)
    # out = wt.artists.interact2D(modes[0], xaxis="energy", yaxis="y")
    # out = wt.artists.interact2D(baseline)
 
    if save:
        p = f"raman_mos2_mode_splitting.SI.png"
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
    run2(save)
