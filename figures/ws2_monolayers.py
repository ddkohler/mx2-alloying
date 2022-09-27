import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
import numpy as np

here = pathlib.Path(__file__).resolve().parent

ws2 = wt.open(here.parent / "data" / "ws2_control.wt5")
hs = wt.open(here.parent / "data" / "heterostructure.wt5")

image = ws2.reflection.image
refl = ws2.reflection.refl.split("yindex", [image.yindex.min() + 115, image.yindex.max() + 115])[1]
pl1 = ws2.pl.face
pl2 = ws2.pl.corner
pl1.convert("eV")
pl2.convert("eV")
raman1 = ws2.raman.face
raman2 = ws2.raman.corner

image.transform("x", "y")
refl.transform("wm", "y")

refl.convert("eV")

if False:  # linescan spectra

    fig, gs = wt.artists.create_figure(
        width="double", nrows=2, cols=[1, 1, "cbar", 1, "cbar"], wspace=0.75
    )

    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(image, cmap="Greys_r", vmin=image.signal.min())

    ax1 = plt.subplot(gs[1, 0])
    ax1.pcolormesh(refl, channel="subtr")
    ax1.set_xlim(1.8, 2.4)

    ax2 = plt.subplot(gs[0,1])
    ax2.pcolormesh(pl1)

    ax3 = plt.subplot(gs[1, 1])
    ax3.pcolormesh(pl2)

    ax4 = plt.subplot(gs[0, 3])
    ax4.pcolormesh(raman1, cmap="magma")

    ax5 = plt.subplot(gs[1, 3])
    ax5.pcolormesh(raman2, cmap="magma")

    plt.show()

if True:  # 1D spectra
    fig, gs = wt.artists.create_figure(
        width="double", nrows=2, cols=[1,1,1], hspace=1
    )  # image, RC, PL, Raman
    ax0 = plt.subplot(gs[0])
    ax0.set_title("h-WS2 ML")
    ax0.imshow(image, cmap="Greys_r", vmin=image.signal.min())
    wt.artists.set_ax_spines(ax0, c="C0")
    
    ax1 = plt.subplot(gs[1])
    ax1.set_title("reflection")
    spectrum = refl.chop("wm", at={"y": [5, "um"]})[0]
    # spectrum.subtr.normalize()
    ax1.plot(spectrum, channel="subtr", label="WS2 pure")

    hs_spectrum = hs.reflection.x20
    hs_spectrum.transform("energy", "ydist")
    hs_spectrum.convert("eV")
    hs_spectrum = hs_spectrum.split("ydist", [15, 20])[1]
    hs_spectrum.create_channel("mean", values=hs_spectrum.contrast[:].mean(axis=1)[:, None])
    # hs_spectrum.mean[:] /= -hs_spectrum.mean.min()
    ax1.plot(hs_spectrum, channel=-1)
    ax1.set_xlim(1.6, 2.5)
    
    ax2 = plt.subplot(gs[2])
    ax2.set_title("PL")
    pl_slice = pl1.chop("wm", at={"x": [10, "um"]})[0]
    pl_slice.signal.normalize()
    ax2.plot(pl_slice, label="WS2 pure")

    hs_pl = hs.pl.proc.chop("energy", at={"x": [12, "um"], "y":[0, "um"]})[0]
    hs_pl.intensity.normalize()
    ax2.plot(hs_pl)

    hs_pl_edge = hs.pl.proc.chop("energy", at={"x": [0, "um"], "y": [18, "um"]})[0]
    hs_pl_edge.intensity.normalize()
    ax2.plot(hs_pl_edge)

    ax3 = plt.subplot(gs[3:])
    ax3.set_title("Raman")

    raman_slice = raman2.chop("wm", at={"x": [10, "um"]})[0]
    hs_raman = hs.raman.raw.chop("energy", at={"x": [12, "um"], "y":[-6, "um"]})[0]
    hs_raman_edge = hs.raman.raw.chop("energy", at={"x": [0, "um"], "y":[12, "um"]})[0]

    # normalize by Si peak
    for data in [raman_slice, hs_raman, hs_raman_edge]:
        idx = np.abs(data.axes[0][:] - 520).argmin()
        data.channels[0][:] /= data.channels[0][idx]

    hs_raman.smooth(5)
    hs_raman_edge.smooth(5)

    ax3.plot(raman_slice)
    ax3.plot(hs_raman)
    ax3.plot(hs_raman_edge)

    ax2.set_xlim(1.6, 2.2)
    ax3.set_xlim(100, 600)
    ax3.set_ylim(0, None)
    
    [ax.set_xlabel(r"$\hbar\omega \ (\mathsf{eV})$") for ax in [ax1, ax2]]
    ax3.set_xlabel(r"$\mathsf{Raman \ Shift} \ (\mathsf{cm}^{-1}) $")
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(c="k", ls="--", lw=0.5)

    # wt.artists.savefig(here / "compare_hs_ws2.png", facecolor="white")
    plt.show()