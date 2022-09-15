import WrightTools as wt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pathlib
import numpy as np

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "heterostructure.wt5"
root = wt.open(data_dir / p)
mpl.style.use(here / "figures.mplstyle")
# root.print_tree(2)

cmap = mpl.cm.get_cmap("magma").copy()
cmap.set_bad("k")


def run(save):
    d = root.raman.proc
    x0 = -6  # 12  # x-slice to examine further (um)

    d_om = root.pl.om
    extent = [d_om.x.min(), d_om.x.max(), d_om.y.min() - 1 , d_om.y.max() - 1]
    img = np.stack([d_om.r, d_om.g, d_om.b], axis=-1)

    # d.transform("energy", "x-y", "x+y")

    if True:
        d0 = d.chop("energy", "y", at={"x": [x0, "um"]})[0]
        d0.smooth((2, 0))
        d0 = d0.split("energy", [440], units="wn")[0]
        d0.intensity.normalize()
        channel="intensity"
    else:  # non-vertical slice (antidiagonal)
        d.smooth((2, 0, 0))
        d0 = d.split("x+y", [1, 3])[1]
        d0 = d0.split("energy", [440], units="wn")[0]
        d0 = d0.split("y", [3])[0]
        d0.create_channel("sum", values=np.nansum(d0.intensity[:], axis=1, keepdims=True))
        d0.sum.normalize()
        d0.sum.log10(-2.)
        print(d0.sum[:,0,:])
        d0.transform("energy", "y")
        channel="sum"

    fig, gs = wt.artists.create_figure(
        width=6.66, nrows=3, margin=[0.8, 0.15, 0.8, 0.8],
        cols=[1, 1.25],
        hspace=1, wspace=0.1
    )

    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(img, extent=extent)
    ax0.set_ylim(-30, 30)

    ax0.vlines([x0], ymin=-30, ymax=30, linewidth=4, color="w", alpha=0.6)
    ax0.set_xlim(x0-30, x0+30)
    ax0.set_xticks([-20, 0, 20])
    ax0.set_yticks([-20, 0, 20])

    ax1 = plt.subplot(gs[0, 1])
    plt.yticks(visible=False)
    ax1.set_yticks([-20, 0, 20])
    # ax1.text(
    #     0.05, 0.05,
    #     r"$x=" + f"{x0}" + r"\ \mathsf{\mu m}$",
    #     transform=ax1.transAxes,
    #     bbox=dict(fc="w", ec="k", boxstyle="square", alpha=0.5)
    # )
    ax1.set_ylim(-30, 30)
    ax1.set_xticks(np.linspace(100, 400, 4))
    y = d0["y"]
    y -= 1
    norm = mpl.colors.LogNorm(vmin=0.01, vmax=1)
    ax1.pcolormesh(d0, channel=channel, cmap=cmap, norm=norm)

    # cax= plt.subplot(gs[0,1])
    ticks = np.linspace(-2, 0, 5)

    divider = make_axes_locatable(ax0)
    hidden_axis = divider.append_axes("top", 0.2, pad=0.1)
    hidden_axis.set_visible(False)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", 0.2, pad=0.1)

    c = wt.artists.plot_colorbar(
        cax,
        cmap=cmap,
        extend="min",
        vlim=[-2, 0],
        ticks=ticks,
        orientation="horizontal",
        label=r"$\mathsf{log(intensity)}$"
    )
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')
    cax.set_facecolor([1,1,1,0])
    
    ax2 = plt.subplot(gs[1:,:])

    yis = [11, 17, 19, 25]  # [21, 19, 17, 15]
    offset = 0
    yticks = [0]

    if True:  # ws2 ref spectrum
        ws2 = wt.open(here.parent / "data" / "ws2_control.wt5").raman.corner

        si_temp = ws2.split("x", [-30])[0].split("wm", [505, 530])[1] # isolate pure substrate
        si_temp.level(0, 0, 5) # remove baseline
        si_peak_sig = si_temp["signal"][:].mean(axis=1).max()

        ws2.signal.normalize()
        # subtract Si Raman
        ws2 = ws2.split("x", [-10, 10])[1]
        ws2.moment(1, 0, moment=0)
        ws2.channels[-1][:] /= ws2.shape[1]
        ws2.transform("wm")
        ax2.plot(ws2, channel=-1, color="k", lw=2, alpha=0.8)
        ax2.fill_between(ws2.wm[:].flatten(), ws2.channels[-1][:].flatten(), offset, color="k", alpha=0.3)
        offset += 1.2 * ws2.channels[-1][:].max()
        ax2.text(250, 0.3, f"pure WS$_2$")

    for i, yi in enumerate(yis):
        spec = d0.chop("energy", at={"y":[yi, "um"]})[0]
        offset_i = max([spec.intensity.max() * 1.2, 0.5])
        ax2.text(250, offset + 0.3 * offset_i, r"$y=$" + str(yi) + r" $\mathsf{\mu m}$")
        spec.intensity[:] += offset
        ax2.fill_between(spec.energy[:], spec.intensity[:], offset, alpha=0.5)
        ax2.plot(spec, channel="intensity")
        if yi == 11:  # special spectrum for highlighting alloy peaks
            for wmi in [199, 378, 397]:
                idx = np.abs(spec.energy.points[:]-wmi).argmin()
                intensity = spec.intensity[idx]
                ax2.text(wmi, intensity, "*", ha="center")
        yticks.append(offset)
        offset += offset_i 

        ax0.scatter(spec.x[:], spec.y[:], zorder=5, s=20, edgecolors="w")

    ax2.set_xlim(125, 440)
    ymin, ymax = ax2.get_ylim()
    ax2.vlines([384, 406], ymin=ymin, ymax=ymax, color="r", lw=1, ls="dashdot")
    ax2.vlines([352, 417.5], ymin=ymin, ymax=ymax, color="b", lw=1, ls="dashdot")

    ax2.set_ylim(ymin, ymax)  # vlines moves the lims, so reset them
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([None for _ in ax2.get_yticklabels()])

    for label, ax in zip("abc", [ax0, ax1, ax2]):
        ax.grid()
        wt.artists.corner_text(label, ax=ax, background_alpha=0.8)

    for ax in [ax1, ax2]:
        ax.set_xlabel(xlabel=r"$\mathsf{Raman \ shift \ (cm^{-1}})$")

    ax0.set_ylabel(r"$y \ (\mu\mathsf{m})$")
    ax0.set_xlabel(r"$x \ (\mu\mathsf{m})$")
    ax2.set_ylabel(r"$\mathsf{Intensity} \ (\mathsf{a.u.})$")

    if save:
        p = f"raman_x_{x0}um.png"
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


