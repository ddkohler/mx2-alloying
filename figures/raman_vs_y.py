import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib
import numpy as np

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "ZYZ543.wt5"
root = wt.open(data_dir / p)

def run(save):
    d = root.raman.proc_raman
    x0 = 12  # x-slice to examine further (um)

    d_om = root.PL.om
    extent = [d_om.x.min(), d_om.x.max(), d_om.y.min(), d_om.y.max()]
    img = np.stack([d_om.r, d_om.g, d_om.b], axis=-1)

    d0 = d.chop("energy", "y", at={"x": [x0, "um"]})[0]
    d0 = d0.split("energy", [440], units="wn")[0]
    d0.smooth((2, 0))
    d0.intensity.normalize()
    d0.intensity.log10()

    fig, gs = wt.artists.create_figure(
        width=6.66, nrows=2, margin=[0.8, 0.15, 0.8, 0.8],
        cols=[1, 1.25],
        aspects=[[[0,0], 0.1],[[1,0], 1]],
        hspace=0.1, wspace=0.1
    )

    ax0 = plt.subplot(gs[1, 0])
    ax0.imshow(img, extent=extent)
    ax0.set_ylim(-30, 30)
    ax0.vlines([x0], ymin=-30, ymax=30, linewidth=4, color="w", alpha=0.6)
    ax0.set_xlim(x0-30, x0+30)
    ax0.set_xticks([0, 20])
    ax0.set_yticks([-20, 0, 20])

    ax1 = plt.subplot(gs[1, 1])
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
    ax1.pcolormesh(d0, channel="intensity", cmap="magma", vmin=d0.intensity.min())

    for ax in [ax0, ax1]:
        ax.grid(color="k", linestyle=":")
    wt.artists.corner_text("a", ax=ax0, background_alpha=0.5)
    wt.artists.corner_text("b", ax=ax1, background_alpha=0.5)

    wt.artists.set_ax_labels(ax1, xlabel=r"$\mathsf{Raman \ shift \ (cm^{-1}})$")
    wt.artists.set_ax_labels(ax0, xlabel=r"$x \ (\mu\mathsf{m})$", ylabel=r"$y \ (\mu\mathsf{m})$")
    cax= plt.subplot(gs[0,1])
    ticks = np.linspace(-2, 0, 5)
    c = wt.artists.plot_colorbar(
        cax,
        cmap="magma",
        vlim=[d0.intensity.min(), d0.intensity.max()],
        ticks=ticks,
        orientation="horizontal",
        label=r"$\mathsf{log(intensity)}$"
    )
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')
    
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


