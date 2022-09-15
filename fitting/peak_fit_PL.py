import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from fitlib import fom
from scipy.optimize import least_squares

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "heterostructure.wt5")
root.print_tree()
screen = wt.open(here.parent / "data" / "clusters.wt5")
screen.print_tree()

verbose = False

# fits
# use Raman data to distinguish different areas
raman = root.raman.proc
pl = root.pl.proc

def fit_screen(screen_name, screen:np.ndarray, data, bounds, verbose=False):
    data.create_variable(screen_name, screen[None, :, :])
    data.print_tree()
    data = data.split(screen_name, [0.5], verbose=verbose)[1]

    # allocate output arrays
    shape = (data.x.size, data.y.size)
    values = {k: np.zeros(shape) for k in ["a1", "r1", "w1", "a2", "r2", "w2", "resid"]}

    # allocate output arrays
    shape = (data.x.size, data.y.size)
    values = {k: np.zeros(shape) for k in ["a1", "r1", "w1", "a2", "r2", "w2", "resid"]}

    for i, j in np.ndindex(shape):
        d_temp = data.chop(
            "energy", 
            at={
                "x":[data.x.points[i], "um"],
                "y":[data.y.points[j], "um"]
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
            0.5 * (bounds[0][2] + bounds[1][2]),
            0.5 * (bounds[0][3] + bounds[1][3]),
            0.05,
            0.1
        ]
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
            values[k][i, j] = val
    for k, val in values.items():
        data.create_channel(k, values=val[None, :, :])
    data.save(here / f"{screen_name}_fitted.wt5")


# --- fit WS2 ---
if True:
    bounds = ([0, 0, 1.85, 1.93, 0.01, 0.01], [np.inf, np.inf, 1.93, 2, 0.2, 0.2])
    # fit_screen("ws2", screen.ws2[:], pl, bounds)
    fit_screen("ws2_edge", screen.junctionb[:], pl, bounds)


if False:
    # --- fit MoS2 ---
    # restrict consideration to MoS2 region
    pl.print_tree()
    pl.create_variable("mos2_core", screen.mos2_core[:][None, :, :])
    pl = pl.split("mos2_core", [0.5], verbose=verbose)[1]


    # allocate output arrays
    shape = (pl.x.size, pl.y.size)
    values = {k: np.zeros(shape) for k in ["a1", "r1", "w1", "a2", "r2", "w2", "resid"]}

    from scipy.optimize import least_squares
    for i, j in np.ndindex(shape):
        d_temp = pl.chop(
            "energy", 
            at={
                "x":[pl.x.points[i], "um"],
                "y":[pl.y.points[j], "um"]
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
            1.82,
            1.87,
            0.1,
            0.05
        ]
        bounds = ([0, 0, 1.75, 1.835, 0.01, 0.01], [np.inf, np.inf, 1.835, 1.87, 0.1, 0.1])
        result = least_squares(
            fom, p0,
            bounds=bounds,
            verbose=False,
            args=[d_temp.energy.points, d_temp.intensity[:]]
        )
        del d_temp
        x = result["x"]
        xs = [x[::2], x[1::2]]  # collate
        out_sorted = sorted(xs, key=lambda l: l[1])
        print(i, j, out_sorted)
        for m, k in enumerate(values.keys()):
            if k == "resid":
                values[k][i,j] = result["cost"]
                continue
            val = out_sorted[m//3][m%3]
            values[k][i, j] = val
    for k, val in values.items():
        pl.create_channel(k, values=val[None, :, :])

    pl.save(here / "mos2_fitted.wt5", overwrite=True)



        