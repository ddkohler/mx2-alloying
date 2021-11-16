import numpy as np
import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt
from fitlib import fom

here = pathlib.Path(__file__).resolve().parent
root = wt.open(here.parent / "data" / "data.wt5")
screen = wt.open(here / "screen.wt5")

verbose = False

if False:
    d = root.PL.proc_PL
    d.print_tree()
    out = wt.artists.interact2D(d, xaxis="energy")
    plt.show()

# fits
# use Raman data to distinguish different areas
raman = root.raman.proc
pl = root.pl.proc

# --- fit WS2 ---
# restrict consideration to WS2 region
pl.create_variable("ws2_core", screen.ws2_core[:][None, :, :])
pl.print_tree()
pl = pl.split("ws2_core", [0.5], verbose=verbose)[1]


# allocate output arrays
shape = (pl.x.size, pl.y.size)
print(shape)
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
        values[k][i, j] = val
for k, val in values.items():
    pl.create_channel(k, values=val[None, :, :])

pl.save(here / "ws2_core_fitted.wt5")



    