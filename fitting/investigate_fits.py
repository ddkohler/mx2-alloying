import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from fitlib import two_res, gauss

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

path = here / "mos2_fitted.wt5"


if True:
    out = wt.artists.interact2D(root.raman.proc, xaxis="energy")
    plt.show()
    1/0


d = wt.open(path)
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

fig, gs = wt.artists.create_figure(width="double", nrows=2, cols=[1, "cbar"] * 2, wspace=1)

for i, chan in enumerate([
    # "area1", "r1", "w1", "area2", 
    "r2",
    # "w2",
    "trion_fraction",
    "rdiff",
    "w_ratio"
]):
    row = i // 2
    col = i % 2
    # print(row, col)
    axi = plt.subplot(gs[row, col * 2])
    wt.artists.corner_text(chan, ax=axi)
    plt.yticks(visible=False)
    if not d[chan].signed:
        d[chan].null = d[chan].min()
    mesh = axi.pcolormesh(d, channel=chan)  # , dynamic_range=True)
    caxi = plt.subplot(gs[row, col * 2 + 1])
    vlim = [d[chan].min(), d[chan].max()]
    if d[chan].signed:
        vlim = mesh.get_clim()  # [d[chan].null - d[chan].major_extent, d[chan].null + d[chan].major_extent]

    wt.artists.plot_colorbar(
        caxi,
        cmap="signed" if d[chan].signed else "default",
        vlim=vlim,
        ticks=np.linspace(*vlim, 6) 
    )

if False:  # scatter plot of strain (A peak position) vs doping (trion fraction
    fig, gs = wt.artists.create_figure(nrows=2)
    ax0 = plt.subplot(gs[0])
    ax0.scatter(d.r2[:].flatten(), d.trion_fraction[:].flatten())
    ax1 = plt.subplot(gs[1])
    ax1.scatter(d.r2[:].flatten(), d.rdiff[:].flatten())

if False:
    d = root.raman.proc
    d.level("intensity", 2, 5)
    out = wt.artists.interact2D(d)
    d.moment(channel="intensity", axis="energy", moment=0)
    # d.level("intensity_energy_moment_0", 2, 5)
    out2 = wt.artists.interact2D(d, channel="intensity_energy_moment_0")

if True:  # check fit quality along several points
    xcoord = 0
    fit = wt.open(path).chop("energy", "y", at={"x":[xcoord, "um"]})[0]
    data = root.pl.proc.chop("energy", "y", at={"x":[xcoord, "um"]})[0]
    ramanx = root.raman.proc.chop("energy", "y", at={"x":[xcoord, "um"]})[0]

    fig, gs = wt.artists.create_figure(cols=[1, 1], default_aspect=2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    offset = 0
    for fi in fit.chop("energy").values():
        # print(fi.y[0])
        if fi.y[0] > 0:
            continue
        di = data.chop("energy", at={"y":[fi.y[0], "um"]}, verbose=False)[0]
        ramanxy = ramanx.chop("energy", at={"y":[fi.y[0], "um"]}, verbose=False)[0]
        ramanxy.smooth(2, verbose=False)
        ax0.plot(di.energy.points, di.intensity[:] + offset)
        ax1.plot(ramanxy.energy.points, ramanxy.intensity[:], color="k", alpha=0.3)  # + offset * 0.1)
        ax0.text(1.7, offset+300, f"y={fi.y[0]}", fontsize=8)
        p = [fi.a1[0], fi.a2[0], fi.r1[0], fi.r2[0], fi.w1[0], fi.w2[0]]
        yfit = two_res(di.energy.points, p) + offset
        yfit_trion = gauss(di.energy.points, p[0::2]) + offset
        yfit_exciton = gauss(di.energy.points, p[1::2]) + offset
        ax0.plot(di.energy.points, yfit, color="k")
        ax0.fill_between(di.energy.points, yfit_trion, offset, color="r", alpha=0.1)
        ax0.fill_between(di.energy.points, yfit_exciton, offset, color="b", alpha=0.1)
        ax0.plot(di.energy.points, yfit, color="k")
        offset += 500


plt.show()
