import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from fitlib import two_res

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

if True:  # process the resulting dataset
    d = wt.open(here / "pl_fitted.wt5")
    # null = np.nanmean(0.5 * d.r1[:] + 0.5 * d.r2[:])
    d.r1.null = np.nanmean(d.r1[:])
    d.r2.null = np.nanmean(d.r2[:])
    d.r1.signed = True
    d.r2.signed = True
    d.create_channel(name="rdiff", values=d.r2[:] - d.r1[:], signed=False)
    # d.create_channel(name="a_ratio", values=d.a2[:]/d.a1[:], signed=False)
    d.create_channel(name="w_ratio", values=d.w2[:]/d.w1[:], signed=False)
    d.create_channel(name="area1", values=d.a1[:] * d.w1[:])
    d.create_channel(name="area2", values=d.a2[:] * d.w2[:])
    d.create_channel(name="trion_fraction", values=d.area1[:] / (d.area1[:] + d.area2[:]))

    fig, gs = wt.artists.create_figure(width="double", nrows=3, cols=[1, "cbar"] * 3, wspace=1)

    for i, chan in enumerate(["area1", "r1", "w1", "area2", "r2", "w2", "trion_fraction", "rdiff", "w_ratio"]):
        row = i // 3
        col = i % 3
        print(row, col)
        axi = plt.subplot(gs[row, col * 2])
        plt.yticks(visible=False)
        axi.pcolormesh(d, channel=chan)
        caxi = plt.subplot(gs[row, col * 2 + 1])
        vlim = [d[chan].null, d[chan].max()]
        if d[chan].signed:
            vlim = [d[chan].null - d[chan].major_extent, d[chan].null + d[chan].major_extent]

        wt.artists.plot_colorbar(
            caxi,
            cmap="signed" if d[chan].signed else "default",
            vlim=vlim,
            ticks=np.linspace(*vlim, 6)
        )

    if True:  # scatter plot of strain (A peak position) vs doping (trion fraction
        fig, gs = wt.artists.create_figure(nrows=2)
        ax0 = plt.subplot(gs[0])
        ax0.scatter(d.r2[:].flatten(), d.trion_fraction[:].flatten())
        ax1 = plt.subplot(gs[1])
        ax1.scatter(d.r2[:].flatten(), d.rdiff[:].flatten())

d = root.raman.raman
out = wt.artists.interact2D(d)

if True:  # check fit quality along several points
    xcoord = -14
    fit = wt.open(here / "pl_fitted.wt5").chop("energy", "y", at={"x":[xcoord, "um"]})[0]
    data = root.PL.PL.chop("energy", "y", at={"x":[xcoord, "um"]})[0]

    plt.figure()
    offset = 0
    for fi in fit.chop("energy").values():
        print(fi.y[0])
        di = data.chop("energy", at={"y":[fi.y[0], "um"]})[0]
        plt.plot(di.energy.points, di.intensity[:] + offset)
        p = [fi.a1[0], fi.a2[0], fi.r1[0], fi.r2[0], fi.w1[0], fi.w2[0]]
        yfit = two_res(di.energy.points, p) + offset
        plt.plot(di.energy.points, yfit, color="k")
        offset += 1000



plt.show()
