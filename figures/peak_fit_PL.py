import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt

here = pathlib.Path(__file__).resolve().parent
data_dir = here.parent / "data"
p = "ZYZ543.wt5"
root = wt.open(data_dir / p)

verbose = False

if False:
    d = root.PL.proc_PL
    d.print_tree()
    out = wt.artists.interact2D(d, xaxis="energy")
    plt.show()


def gauss(x, p):
    a, r, w = p
    s = w / 2.35
    return a * np.exp(-(x-r)**2 / (2**0.5 * s)**2)


def lorentzian(x, p):
    a, r, w = p
    return a * w**2 / ((x-r)**2 + w**2)


def two_res(x, p):
    a1, a2, r1, r2, w1, w2 = p
    out = gauss(x, [a1, r1, w1])
    out += gauss(x, [a2, r2, w2])
    return out


def fom(p, w, y):
    fit = two_res(w, p)
    return y - fit


if False:  # fit WS2 PL data
    raman = root.raman.proc_raman
    d = root.PL.proc_PL
    # restrict consideration to WS2 region
    d_temp = raman.chop("x", "y", at={"energy": [350, "wn"]}, verbose=verbose)[0]
    d_temp.leveled.normalize()
    separator = d_temp.leveled.points  # note raman data y axis differs from pl data by 1 pixel
    d.create_variable("filter", values=np.roll(separator, -1, axis=1)[None, :, :])  # shifting because Raman data is shifted
    if False:
        d.create_channel("separator", values=separator[None, :, :])
        d.print_tree()
        out2 = wt.artists.interact2D(d, channel="intensity")
        out1 = wt.artists.interact2D(d, channel="separator")
        plt.show()
        1/0

    d = d.split("filter", [0.1], verbose=verbose)[1]
    del d_temp

    # allocate output arrays
    shape = (d.x.size, d.y.size)
    print(shape)
    values = {k: np.zeros(shape) for k in ["a1", "r1", "w1", "a2", "r2", "w2", "resid"]}

    from scipy.optimize import least_squares
    for i, j in np.ndindex(shape):
        d_temp = d.chop(
            "energy", 
            at={
                "x":[d.x.points[i], "um"],
                "y":[d.y.points[j], "um"]
            },
            verbose=False
        )[0]
        if np.all(np.isnan(d_temp.intensity[:])):
            for k in values.keys():
                # print(i, j, k, values[k][i,j])
                values[k][i,j] = np.nan
            continue
        p0 = [
            d_temp.intensity.max() / 2,
            d_temp.intensity.max() / 2,
            1.86,
            1.95,
            0.05,
            0.1
        ]
        bounds = ([0, 0, 1.85, 1.91, 0.01, 0.01], [np.inf, np.inf, 1.91, 2, 0.4, 0.4])
        result = least_squares(
            fom, p0,
            bounds=bounds,
            verbose=False,
            args=[d_temp.energy.points, d_temp.intensity[:]]
        )
        x = result["x"]
        xs = [x[::2], x[1::2]]  # collate
        out_sorted = sorted(xs, key=lambda l: l[1])
        print(i, j, out_sorted)
        for m, k in enumerate(values.keys()):
            if k == "resid":
                values[k][i,j] = result["cost"]
                continue
            val = out_sorted[m//3][m%3]
            values[k][i,j] = val
    for k, val in values.items():
        d.create_channel(k, values=val[None, :, :])

    d.save(here / "pl_fitted.wt5")

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

d = root.raman.proc_raman
out = wt.artists.interact2D(d)

if True:  # check fit quality along several points
    xcoord = -14
    fit = wt.open(here / "pl_fitted.wt5").chop("energy", "y", at={"x":[xcoord, "um"]})[0]
    data = root.PL.proc_PL.chop("energy", "y", at={"x":[xcoord, "um"]})[0]

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


    