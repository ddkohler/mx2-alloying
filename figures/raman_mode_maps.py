import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt


def run(save):
    here = pathlib.Path(__file__).resolve().parent
    data_dir = here.parent / "data"
    p = "heterostructure.wt5"
    root = wt.open(data_dir / p)
    d = root.raman.proc

    # define masks for MoS2, WS2, and Si
    d_temp = d.chop("x", "y", at={"energy": [383, "wn"]}, verbose=False)[0]
    d_temp.intensity.normalize()
    separator = d_temp.intensity[:]
    MX2_area = separator > 0.2
    WS2_area = separator > 0.36
    MoS2_area = MX2_area & ~WS2_area

    # frequencies picked by hand to represent peak transitions
    WS2_freqs= [174, 198, 211, 350, 378, 397, 416]
    MoS2_freqs = [383, 404, 408]
    Si_freqs = [520]

    row_names = [r"WS$_2$", r"MoS$_2$", "Si"]

    fig, gs = wt.artists.create_figure(
        width="dissertation", nrows=3, cols=[1] * len(WS2_freqs) + ["cbar"], hspace=0.5
    )
    for row, (freqs, mask) in enumerate(
        zip(
            [WS2_freqs, MoS2_freqs, Si_freqs],
            [None, WS2_area, None]  # MoS2 masks WS2 for better contrast
    )):
        for col, freq in enumerate(freqs):
            ax = plt.subplot(gs[row, col])
            if col==0:
                ax.set_ylabel(row_names[row])
            chop = d.chop("x", "y", at={"energy": [freq, "wn"]}, verbose=False)[0]
            z = chop.leveled[:]
            if mask is not None:
                z[mask] = 0
            ax.pcolormesh(chop.x.points, chop.y.points, z.T, cmap="magma", vmin=0)
            ax.set_title(str(freq) + " wn")
            plt.yticks(visible=False)
            plt.xticks(visible=False)
    cax = plt.subplot(gs[:, -1])
    wt.artists.plot_colorbar(cax, cmap="magma", label="Raman intensity (norm.)")

    if save:
        p = f"raman_mode_maps.png"
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


