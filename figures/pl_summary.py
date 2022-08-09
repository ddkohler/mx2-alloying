import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "data.wt5"
root = wt.open(data_dir / p)
# root.print_tree(2)

grid_kwargs = {
    "ls": ":",
    "lw": 0.5,
    "c": "k"
}

eV_label = r"$\hbar\omega \ (\mathsf{eV})$"
x_label = r"$x \ (\mathsf{\mu m})$"
y_label = r"$y \ (\mathsf{\mu m})$"

x0 = -12  # x-slice to examine further (um)
ylims = [-20, -8]  # y rangle that has the interface (at x0)

def run(save):
    import matplotlib.patches as patches

    d = root.pl.proc

    # d.smooth((5, 0, 0))
    screen = root.raman.proc.chop("x", "y", at={"energy":[350, "wn"]})[0]
    screen.intensity.normalize()

    fig, gs = wt.artists.create_figure(
        width="double", nrows=2,
        cols=[1, 0.1, 1, 1],
        margin=[1, 0.5, 1, 0.25],
        hspace=0.2,
        aspects=[[[0,0], 0.1],[[1,0], 1]]
    )
    ax_image = plt.subplot(gs[:,0])
    img = mplimg.imread(here.parent / "data" / "characterization" / "microscope1 - enhanced contrast.jpg")
    img = ax_image.imshow(img.transpose(1,0,2))
    # scale bar; 92 um ~ 302 pixels -> 3.28 pixels / um
    scalebar = AnchoredSizeBar(
        ax_image.transData,
        328, r'100 $\mathsf{\mu}$m', 'lower right', 
        pad=0.3,
        color='black',
        frameon=False,
        size_vertical=10,
        fill_bar=True
    )
    ax_image.add_artist(scalebar)
    # ax_image.plot([400, 400 + 3.28 * 100], [])

    ax_spatial = plt.subplot(gs[1, 2])

    # mask
    arr = d.intensity[:]
    arr_collapse = np.mean(arr, axis=0)

    mask = np.ones(arr_collapse.shape)
    mask[arr_collapse < arr_collapse.max() * 0.05] = 0
    nanmask = mask.copy()
    nanmask[nanmask == 0] = np.nan

    # # datasets to plot
    pl_area = d.intensity[:].mean(axis=0)
    pl_color = d.intensity_energy_moment_1[0] * nanmask
    pl_color[pl_color < 1.82] = 1.82

    # if x, y are 1D, use z.T; if x, y are 2D, use z
    if False:  # for testing alignment of reflection, PL images
        # triangle boundary
        # ax0.contour(d.axes[0].points, d.axes[1].points, mask.T, levels=np.array([0.5]), alpha=0.5)
        # junction
        # ax0.contour(d.axes[0].points, d.axes[1].points, pl_color.T, levels=np.array([1.91]), alpha=0.3)
        ax_spatial.contour(d.axes[0].points, d.axes[1].points, pl_color.T, levels=np.array([1.91]), alpha=0.3)

    pcolormesh = ax_spatial.pcolormesh(d.axes[0].points, d.axes[1].points, pl_color.T, cmap="rainbow_r", shading="auto")
    ax_spatial.contour(
        screen, channel="intensity",
        levels=np.array([-1, 0.5, 2]), colors="w", linewidths=2, alpha=1
    )
    ax_spatial.text(-12, 3, r"$\mathsf{WS}_2$", color="k", fontsize=20) # , path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    ax_spatial.text(-40, -20, r"$\mathsf{MoS}_2$", color="k", fontsize=20) # , path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    rect1 = patches.Rectangle(
        (x0 - 1, ylims[0]-1), 2, ylims[1] - ylims[0] + 2, linewidth=2, edgecolor='k', facecolor='k', fill=False, zorder=9
    )
    ax_spatial.add_patch(rect1)

    ax_spatial.set_ylim(-40, 60)
    ax_spatial.set_xlim(-50, 50)

    # plt.yticks(visible=False)
    ax_intensity = ax_spatial.inset_axes([0.7, 0.7, 0.3, 0.3])
    ax_intensity.pcolormesh(d.axes[0].points, d.axes[1].points, pl_area.T, cmap="gist_gray", shading="auto")

    ax_spatial_cb = plt.subplot(gs[0, 2])
    wt.artists.plot_colorbar(
        cax=ax_spatial_cb,
        cmap="rainbow_r",
        orientation="horizontal",
        vlim=[np.nanmin(pl_color), np.nanmax(pl_color)],
        ticks=np.linspace(1.8, 1.92, 7),
        label="PL " + r"$\langle\hbar\omega\rangle \ (\mathsf{eV})$"
    )

    crossover = d.chop("energy", "y", at={"x":[x0, 'um']})[0]
    crossover.transform("energy", "y")
    crossover = crossover.split("x", [-30, 40])[1]
    crossover.intensity[:] *= nanmask[d.x.points==x0]
    crossover.intensity.normalize()
    if False:
        ax2 = plt.subplot(gs[1, 1])
        # avoid showing gray as underflow; use white
        cmap = wt.artists.colormaps["default"]
        cmap.set_under([1,1,1])
        crossover.print_tree()
        ax2.pcolormesh(crossover)# , cmap=cmap)
        ax2.text(0.35, 0.87, r"$x=" + str(x0) + r"\ \mathsf{\mu m}$", transform=ax2.transAxes, fontsize=16)
        ys = [-20, -8]
        rect2 = patches.Rectangle(
            (1.66, ys[0] - 1), 0.4, ys[1] - ys[0], linewidth=1, edgecolor='k',facecolor='k', fill=False
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

    ax3 = plt.subplot(gs[:, 3])
    crossover.intensity[:] /= np.nanmax(crossover.intensity[:], axis=0)
    spectra = crossover.split("y", ylims, units="um")[1]
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

    # add in control spectrum
    control = wt.open(data_dir / "zyz-554.wt5").pl.face
    control = control.chop("wm", at={"x":[0, "um"]})[0]
    control.convert("eV")
    control.signal.normalize()
    ax3.plot(control.wm[:], control.signal[:] + len(spectra), "k", lw=2)
    ax3.text(1.8, len(spectra) + 0.4, r"$\mathsf{pure \ WS}_2$", fontsize=16)
    ax3.set_xlim(1.65, 2.06)


    for c_ax in [ax_spatial_cb]:
        c_ax.xaxis.set_label_position('top')
        c_ax.xaxis.set_ticks_position('top')

    ax_spatial.set_xticks(np.linspace(-25, 25, 3))
    ax_spatial.set_yticks(np.linspace(-25, 25, 3))
    ax_spatial.set_ylim(d.y.min(), d.y.max())
    ax_intensity.set_xticklabels([None for i in ax_intensity.get_xticklabels()])
    ax_intensity.set_yticklabels([None for i in ax_intensity.get_yticklabels()])
    # ax_image.set_xticklabels([None for i in ax_image.get_xticklabels()])
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    # ax2.set_yticks([i for i in ax_spatial.get_yticks()])
    # ax2.set_yticklabels([None for i in ax2.get_yticklabels()])
    # ax2.set_ylim(*ax_spatial.get_ylim())
    ax3.set_yticks([i for i in range(len(spectra) + 1)])
    ax3.set_yticklabels([None for i in ax3.get_yticks()])
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    for ax in [ax_spatial, ax3]:
        ax.grid(**grid_kwargs)

    wt.artists.set_ax_labels(ax_spatial, ylabel=y_label, xlabel=x_label)
    # wt.artists.set_ax_labels(ax2, xlabel=eV_label)
    wt.artists.set_ax_labels(ax3, xlabel=eV_label, ylabel="PL intensity (norm.)")

    for i, ax in enumerate([ax_image, ax_spatial, ax3]):
        wt.artists.corner_text("abcd"[i], ax=ax, background_alpha=0.3)

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
