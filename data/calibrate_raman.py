import pathlib
import WrightTools as wt
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


here = pathlib.Path(__file__).resolve().parent
root = wt.open(here / "data.wt5")
raman = root.raman.raw.split("energy", [450])[1].split("x", [-20])[0].split("y", [20])[1]
raman.level(0, 0, 20)

if False:
    out = wt.artists.interact2D(raman)
    plt.show()

y = raman.intensity[:].mean(axis=(1,2))
std_y = raman.intensity[:].std(axis=(1,2))
x = raman.energy.points


def lorentz(x, p):
    a, w, x0 = p
    return a * w**2 / ((x-x0)**2 + w**2)


def gauss(x, p):
    a, w, x0 = p
    return a * np.exp(-(x-x0)**2 / (2 * w**2))

def fit(x, p):
    p0, p1 = p[:3], p[3:]
    return lorentz(x, p0) + gauss(x, p1)


def err(p, x, y):
    guess = fit(x, p)
    return y - guess


p0 = [140, 5, 520, 10, 5, 524]
bounds =(
    [0, 0, 519, 0, 0, 520],
    [np.inf, np.inf, 522, np.inf, np.inf, 525]
)

result = least_squares(err, p0, args=[x, y], bounds=bounds)
print(f"result {result['x']}")
y_fit = fit(x, result["x"])

fig, gs = wt.artists.create_figure()
ax = plt.subplot(gs[0])
#ax.plot(x, y)
ax.fill_between(x, y-std_y, y+std_y, alpha=0.5)
ax.fill_between(x, -std_y, std_y, alpha=0.5)
ax.plot(x, y_fit, color="k")
ax.plot(x, y-y_fit, color="r")

plt.show()

# in raw data, Si Raman peak is at ~519.9 +/- 0.1, but should be 520.7 (+/- 0.5).
# Apply correction to proc.


