import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib

here = pathlib.Path(__file__).resolve().parent
um_per_pixel = 0.6  # 10x

d = wt.data.from_Solis(here / "10x image lamp.asc").split("xindex", [740, 1900])[1]  # .split("yindex", [750, 1750])[1]
d.create_variable("x", values=d.xindex[:] * um_per_pixel, units="um")
d.create_variable("y", values=d.yindex[:] * um_per_pixel, units="um")

d.transform("x", "y")

fig, gs = wt.artists.create_figure()
ax0 = plt.subplot(gs[0])
ax0.imshow(d, cmap="Greys_r", vmin=1e5)
ax0.set_xlabel(d.x.label)
ax0.set_ylabel(d.y.label)
plt.show()
