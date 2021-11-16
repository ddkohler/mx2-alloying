import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from fitlib import fom

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")

verbose = False

if False:
    d = root.PL.proc_PL
    d.print_tree()
    out = wt.artists.interact2D(d, xaxis="energy")
    plt.show()

if True:  # fits
    raman = root.raman.raman
    d = root.PL.PL
    # restrict consideration to WS2 region
    # d_temp = raman.chop("x", "y", at={"energy": [350, "wn"]}, verbose=verbose)[0]
    d_temp = raman.split("energy", [150])[0]
    d_temp.level("intensity", 2, 5)
    d_temp.moment(channel="intensity", axis="energy", moment=0)
    separator = d_temp.intensity_energy_moment_0
    separator.normalize()
    # note raman data y axis differs from pl data by 1 pixel
    # rolling because Raman data is shifted from PL data
    d.create_variable("filter", values=np.roll(separator[0], -1, axis=1)[None, :, :])
    if False:
        d.create_channel("separator", values=separator[None, :, :])
        out2 = wt.artists.interact2D(d, channel="intensity")
        out1 = wt.artists.interact2D(d, channel="separator")
        plt.show()
        1/0

    # --- fit WS2 ---
    d = d.split("filter", [0.5], verbose=verbose)[1]
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



    