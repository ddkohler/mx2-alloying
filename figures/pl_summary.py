import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "ZYZ543.wt5"
root = wt.open(data_dir / p)

eV_label = r"$\hbar\omega \ (\mathsf{eV})$"
x_label = r"$x \ (\mathsf{\mu m})$"
y_label = r"$y \ (\mathsf{\mu m})$"


def run(save):
    import matplotlib.patches as patches

    d = root.PL.proc_PL
    d.level("intensity", 2, 2)
    d.smooth((5, 0, 0))
    x0 = 12  # x-slice to examine further (um)
    screen = root.raman.proc_raman.chop("x", "y", at={"energy":[350, "wn"]})[0]
    screen.intensity.normalize()

    # d_om = root.PL.om
    # d_om.print_tree()
    
    # img = np.stack([d_om.r, d_om.g, d_om.b], axis=-1)
    # extent = [d_om.x.min(), d_om.x.max(), d_om.y.min(), d_om.y.max()]

    fig, gs = wt.artists.create_figure(
        width="double", nrows=2,
        cols=[1, 1, 1],
        margin=[1, 0.5, 1, 1],
        hspace=0.2,
        aspects=[[[0,0], 0.1],[[1,0], 1]]
    )
    # ax0 = plt.subplot(gs[1, 0])
    # ax0.imshow(img, extent=extent)
    # ax0.set_title("Reflection \n (Enh. Contrast)")

    ax1 = plt.subplot(gs[1, 0])
    # mask
    arr = d.intensity[:]
    arr_collapse = np.mean(arr, axis=0)

    mask = np.ones(arr_collapse.shape)
    mask[arr_collapse < arr_collapse.max() * 0.04] = 0
    nanmask = mask.copy()
    nanmask[nanmask == 0] = np.nan

    # datasets to plot
    pl_area = d.intensity[:].mean(axis=0)
    pl_color = d.intensity_energy_moment_1[0] * nanmask
    pl_color[pl_color < 1.81] = 1.81

    # if x, y are 1D, use z.T; if x, y are 2D, use z
    if False:  # for testing alignment of reflection, PL images
        # triangle boundary
        # ax0.contour(d.axes[0].points, d.axes[1].points, mask.T, levels=np.array([0.5]), alpha=0.5)
        # junction
        # ax0.contour(d.axes[0].points, d.axes[1].points, pl_color.T, levels=np.array([1.91]), alpha=0.3)
        ax1.contour(d.axes[0].points, d.axes[1].points, pl_color.T, levels=np.array([1.91]), alpha=0.3)

    pcolormesh = ax1.pcolormesh(d.axes[0].points, d.axes[1].points, pl_color.T, cmap="rainbow_r")
    ax1.contour(
        screen, channel="intensity",
        # d.axes[0].points, d.axes[1].points-2, d.screen[0].T,
        levels=np.array([0, 0.5, 1]), colors="w", linewidths=2, alpha=1
    )
    ax1.text(-10, 10, r"$\mathsf{WS}_2$", color="w", fontsize=16)
    ax1.text(-3, 30, r"$\mathsf{MoS}_2$", color="w", fontsize=16)
    rect1 = patches.Rectangle(
        (x0 - 1, -29), 2, 70, linewidth=1, edgecolor='k', facecolor='k', fill=False
    )
    ax1.add_patch(rect1)

    ax1.set_ylim(-40, 60)
    ax1.set_xlim(-50, 50)

    # plt.yticks(visible=False)
    ax_intensity = ax1.inset_axes([0.7, 0.7, 0.3, 0.3])  # plt.subplot(gs[0,3], sharex=ax0, sharey=ax0)
    ax_intensity.pcolormesh(d.axes[0].points, d.axes[1].points, pl_area.T, cmap="gist_gray")

    ax1cb = plt.subplot(gs[0, 0])
    wt.artists.plot_colorbar(
        cax=ax1cb,
        cmap="rainbow_r",
        orientation="horizontal",
        clim=[np.nanmin(pl_color), np.nanmax(pl_color)],
        ticks=np.linspace(1.8, 1.92, 7),
        label="PL " + r"$\langle\hbar\omega\rangle \ (\mathsf{eV})$"
    )

    crossover = d.chop("energy", "y", at={"x":[x0, 'um']})[0]
    crossover = crossover.split("x", [-30, 40])[1]
    crossover.intensity[:] *= nanmask[d.x.points==x0]
    crossover.intensity.normalize()
    crossover.print_tree()

    ax2 = plt.subplot(gs[1, 1])
    ax2.pcolormesh(crossover)
    ax2.text(0.4, 0.85, r"$x=12 \ \mathsf{\mu m}$", transform=ax2.transAxes, fontsize=16)
    rect2 = patches.Rectangle(
        (1.66, -21), 0.4, 10, linewidth=1,edgecolor='k',facecolor='k', fill=False
    )
    ax2.add_patch(rect2)
    # plt.ylim(-30, 40)

    ax2cb = plt.subplot(gs[0, 1])
    wt.artists.plot_colorbar(
        cax=ax2cb, cmap=wt.artists.colormaps["default"],
        orientation="horizontal",
        vlim=[0, 1],
        ticks=np.linspace(0, 1, 6),
        label="PL intensity (a.u.)"
    )

    ax3 = plt.subplot(gs[1, 2])
    crossover.intensity[:] /= np.nanmax(crossover.intensity[:], axis=0)
    spectra = crossover.split("y", [-20, -10], units="um")[1]
    spectra = spectra.chop("energy")
    centers = np.array([spec.intensity_energy_moment_1[0] for spec in spectra.values()])
    centers -= np.nanmin(pl_color)
    centers /= np.nanmax(pl_color) - np.nanmin(pl_color)
    cmap = pcolormesh.get_cmap()
    colors = cmap(centers)

    for i, spec in enumerate(spectra.values()):
        spec.smooth(2)
        ax3.plot(
            spec.energy[:], spec.intensity[:] + i,
            color=colors[i],
            alpha=1, linewidth=2, label=int(spec.y[0]))
        text = str(int(spec.y[0])) + r"\ \mu\mathsf{m}$"
        if i==len(spectra)-1:
            text = r"$y=" + text
        else:
            text = r"$" + text
        ax3.text(1.65, i + 0.1, text, fontsize=16)

    for c_ax in [ax1cb, ax2cb]:
        c_ax.xaxis.set_label_position('top')
        c_ax.xaxis.set_ticks_position('top')

    ax1.set_xticks(np.linspace(-25, 25, 3))
    ax1.set_yticks(np.linspace(-25, 25, 3))
    ax_intensity.set_xticklabels([None for i in ax_intensity.get_xticklabels()])
    ax_intensity.set_yticklabels([None for i in ax_intensity.get_yticklabels()])
    ax2.set_yticklabels([None for i in ax2.get_yticklabels()])
    ax3.set_yticks([i for i in range(len(spectra))])
    ax3.set_yticklabels([None for i in range(len(spectra))])
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    for ax in [ax1, ax2, ax3]:
        ax.grid()

    wt.artists.set_ax_labels(ax1, ylabel=y_label, xlabel=x_label)
    wt.artists.set_ax_labels(ax2, xlabel=eV_label)
    wt.artists.set_ax_labels(ax3, xlabel=eV_label, ylabel="PL intensity (norm.)")

    for i, ax in enumerate([ax1, ax2, ax3]):
        wt.artists.corner_text("abcd"[i], ax=ax)

    if save:
        p = "pl_map_summary.png"
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
