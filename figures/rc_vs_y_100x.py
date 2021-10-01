import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "ZYZ543.wt5"

def run(save):
    wt5 = data_dir / "reflection_microspectroscopy" / "reflection.wt5"    
    reflection = wt.open(wt5)
    img = reflection.image
    spectrum = reflection.spectrum

    fig, gs = wt.artists.create_figure(
        width=6.66, nrows=2, margin=[0.8, 0.15, 0.8, 0.8],
        cols=[1, 1.25],
        aspects=[[[0,0], 0.1],[[1,0], 1]],
        hspace=0.1, wspace=0.1
    )


    ax0 = fig.add_subplot(gs[1,0])
    ax0.set_facecolor((0,0,0,0))
    ax0.pcolormesh(img, channel="signal", cmap="gist_gray")
    ax0.set_ylim(-30, 30)
    ax0.set_xlim(-30, 30)
    ax1 = fig.add_subplot(gs[1,1])
    # ax1.set_title(r"$x=0$")
    ax1.pcolormesh(spectrum, channel="contrast")
    ax1.vlines([1e7/532/8065.5], ymin=img.y.min(), ymax=img.y.max(), linestyle='-', color="g", linewidth=3, alpha=0.7)
    y = spectrum.y[:][0]
    valid = y > -23
    for x, linestyle in zip([spectrum.ares[:][0], spectrum.bres[:][0]], ["-", "-"]):
        ax1.plot(
            x[valid], y[valid],
            color="k", linewidth=1, alpha=0.3, linestyle=linestyle
        )
    ax0.vlines(
        [0], ymin=img.y.min(), ymax=img.y.max(), linestyle="--", color="k"
    )

    ax0.grid(axis="y", color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax1.grid(axis="y", color="k", linestyle="--", alpha=0.5, linewidth=1)
    ax1.grid(axis="x", color="k", linestyle='-', alpha=0.3)
    wt.artists.set_ax_labels(ax0, xlabel=r"$x \ (\mu\mathsf{m})$", ylabel=r"$y \ (\mu\mathsf{m})$")
    wt.artists.set_ax_labels(ax1, xlabel=r"$\hbar\omega \ (\mathsf{eV})$")
    ax1.set_yticklabels(["" for i in ax1.get_yticklabels()])
    cax = fig.add_subplot(gs[0, 1])

    for i, ax in enumerate([ax0, ax1]):
        wt.artists.corner_text(
            "ab"[i],
            ax=ax,
            # distance=.03,
            # fontsize=14,
            background_alpha=.8,
            bbox=True,
            factor=200,
        )

    vmag = max(spectrum.contrast.min(), spectrum.contrast.max())
    wt.artists.plot_colorbar(
        cax,
        cmap="signed",
        ticks=np.linspace(-0.3, 0.3, 7),
        vlim=[-vmag, vmag],
        orientation="horizontal",
        decimals=1,
        label=r"$\left(R-R_0\right) / R_0$"
    )
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')

    # save
    if save:
        p = "junction_reflection_contrast.png"
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

