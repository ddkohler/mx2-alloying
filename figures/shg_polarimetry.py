import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os


def run(save):
    """shg polarization summary image"""
    here = pathlib.Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    p = "heterostructure.wt5"
    root = wt.open(data_dir / p)
    shg_pol = root.shg  # wt.open(data_dir / "polarization.wt5")
    verbose=False
    from scipy.interpolate import interp1d

    def raw_angle_to_physical(deg):
        """ 
        Relate recorded angle to the image axes.
        Raw number is off by an offset
        0 deg corresponds to H-pol in the lab frame, which is V-pol for our image in this figure
        """
        offset = 49.
        return deg - offset

    slit = shg_pol.imgs.slit # .split("yindex", ylims, verbose=verbose)[1]
    lamp = shg_pol.imgs.lamp # .split("yindex", ylims, verbose=verbose)[1]
    pump = shg_pol.imgs.pump # .split("yindex", ylims, verbose=verbose)[1]
    temp = shg_pol.polarization

    slit.signal.normalize()
    for data in [slit, lamp, pump]:
        data.transform("ydistance", "xdistance")

    shg_pol.polarization.transform("yindex", "angle")

    temp.signal_xindex_moment_0.normalize()

    fig, gs = wt.artists.create_figure(
        width="dissertation",
        nrows = 4,
        cols = [1, "cbar"],
        default_aspect=1/6.
    )

    ax1 = plt.subplot(gs[0, 0])
    plt.xticks(visible=False)
    ax2 = plt.subplot(gs[1:, 0], sharex=ax1)

    ax1.pcolormesh(
        lamp,
        cmap="gist_gray",
        vmax=shg_pol.imgs.lamp.signal.max() * 0.8,
        vmin=shg_pol.imgs.lamp.signal.max() * 0.68
    )
    pump_cm = ListedColormap(np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0.2]
    ]))

    pump.signal[:] /= pump.signal_xindex_moment_0[:]
    pump.signal.normalize()
    ax1.contourf(
        pump,
        levels=np.array([0.5, 1]), cmap=pump_cm
    )
    # ax1.contourf(
    #     slit.yindex[0, :], slit.xindex[:, 0], slit.signal[:],
    #     levels=np.array([0.5, 1]), cmap=slit_cm
    # )

    angle = temp["angle"]
    angle[:] = raw_angle_to_physical(angle[:])
    temp.transform("ydistance", "angle")
    ax2.pcolormesh(temp, channel="signal_xindex_moment_0")
    ax2.set_yticks(np.linspace(-30, 270, 6))
    for ax in [ax1, ax2]:
        ax.grid(color="k", linestyle=":")

    # ax1.set_xlim(200, 950)
    wt.artists.set_ax_labels(ax=ax2, xlabel=r"$x \ \left(\mu \mathsf{m}\right)$", ylabel=r"$\theta_{\mathsf{out}} = \theta_{\mathsf{in}} \ (\mathsf{deg})$")
    wt.artists.set_ax_labels(ax=ax1, ylabel=r"$y \ \left(\mu \mathsf{m}\right)$")
    cax = plt.subplot(gs[1:, 1])
    wt.artists.plot_colorbar(cax=cax, label="SHG Intensity (a.u.)")
    wt.artists.corner_text("a", ax=ax1)
    wt.artists.corner_text("b", ax=ax2)

    if save:
        wt.artists.savefig(here / "polarization_summary.SI.png")
    else:
        plt.show()



def run2(save):
    """shg polarization summary image - polished"""
    here = pathlib.Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    p = "heterostructure.wt5"
    root = wt.open(data_dir / p)
    shg_pol = root.shg
    verbose = False
    from scipy.interpolate import interp1d

    def raw_angle_to_physical(deg):
        """ 
        Relate recorded angle to the image axes.
        Raw number is off by an offset
        0 deg corresponds to H-pol in the lab frame, which is V-pol for our image in this figure
        """
        offset = 49.
        return deg - offset

    # ylims = [200, 950]
    ylims = [750, 950]
    slit = shg_pol.imgs.slit.split("yindex", ylims, verbose=verbose)[1]
    lamp = shg_pol.imgs.lamp.split("yindex", ylims, verbose=verbose)[1]
    pump = shg_pol.imgs.pump.split("yindex", ylims, verbose=verbose)[1]

    slit.signal.normalize()
    for data in [slit, lamp, pump]:
        print(data.variables)
        data.transform("ydistance", "xdistance")
    
    shg_pol.polarization.transform("ydistance", "angle")

    temp = shg_pol.polarization.split("yindex", ylims)[1]
    temp.signal_xindex_moment_0.normalize()
    temp.moment("angle", channel="signal_xindex_moment_0", moment=0)  # signal at all angles

    # center images
    for d in [temp, lamp, pump]:
        d.print_tree()
        xdist = d["xdistance"]
        ydist = d["ydistance"]
        xdist[:] -= xdist[:].mean()
        ydist[:] -= ydist[:].mean()

    fig, gs = wt.artists.create_figure(
        width=6.66,
        nrows = 2,
        cols = [1],
        default_aspect = 100 / (ylims[1] - ylims[0]),
        margin=[0.8, 0.15, 0.8, 0.8]
    )

    ax1 = plt.subplot(gs[0, :])
    plt.xticks(visible=False)
    ax2 = plt.subplot(gs[1, :], sharex=ax1)
    from mpl_toolkits.axes_grid1 import inset_locator
    import matplotlib as mpl
    ax3 = inset_locator.inset_axes(
        ax2,
        "35%",
        "60%",
        axes_class = mpl.projections.get_projection_class('polar'),
        loc="lower right"
    )
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax1.pcolormesh(
        lamp, cmap="gist_gray", vmax=shg_pol.imgs.lamp.signal.max() * 0.8, vmin=shg_pol.imgs.lamp.signal.max() * 0.68
    )
    pump_cm = ListedColormap(np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0.2]
    ]))

    pump.signal[:] /= pump.signal_xindex_moment_0[:]
    pump.signal.normalize()
    ax1.contourf(
        pump.ydistance[0, :], pump.xdistance[:, 0], pump.signal[:],
        levels=np.array([0.5, 1]), cmap=pump_cm
    )
    ax1.text(0.38, 0.65, r"MoS$_2$", transform=ax1.transAxes, fontsize=16)
    ax1.text(0.7, 0.23, r"WS$_2$", transform=ax1.transAxes, fontsize=16)

    angle = temp["angle"]
    angle[:] = raw_angle_to_physical(angle[:])
    if True:
        yi = temp.signal_xindex_moment_0[0, :, 6]
        yi /= yi.max()
        ax2.plot(temp.ydistance.points, yi, linewidth=3, color="k")
        ax2.set_ylabel("SHG intensity (a.u.)", fontsize=18)
    else:
        temp.transform("ydistance", "angle")
        ax2.pcolormesh(temp, channel="signal_xindex_moment_0")
        ax2.set_ylabel(r"$\theta_{\mathsf{out}} = \theta_{\mathsf{in}} \ (\mathsf{deg})$")
    ax2.set_xlabel(r"$x \ (\mathsf{\mu m})$", fontsize=18)
    ax1.set_ylabel(r"$y \ (\mathsf{\mu m})$", fontsize=18)
    wt.artists.corner_text("a", ax=ax1)
    wt.artists.corner_text("b", ax=ax2)

    temp.print_tree()
    theta = temp.angle.points * np.pi / 180
    sim_theta = np.linspace(-np.pi, np.pi, 101)

    from figlib import colors

    ylim = ax2.get_ylim()
    for label, x_range, color in zip(
        ["edge WS2", "WS2", "MoS2"],
        [[3, 6], [10, 20], [-10, 0]],
        [colors[4], colors[5], colors[2]]
    ):
        split = temp.split("ydistance", x_range)[1]
        y = split.signal_xindex_moment_0[0].mean(axis=0)
        y /= y.mean()
        ax2.fill_betweenx(ylim, *x_range, color=color, alpha=1)
        ax3.plot(theta, y, color=color, alpha=1, lw=2)
    ax2.set_ylim(*ylim)

    ax3.plot(sim_theta, 2 * np.cos(3 * sim_theta - 30 * np.pi / 180)**2, color="k")

    for ax in [ax1, ax2, ax3]:
        ax.grid(color="k", linestyle=":")

    if save:
        wt.artists.savefig(here / "polarization_summary.main.png")
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

