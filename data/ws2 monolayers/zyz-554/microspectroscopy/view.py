import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib

here = pathlib.Path(__file__).resolve().parent
um_per_pixel = 0.3  # 20x

ref = wt.open(here / "ref.wt5")
blank = wt.open(here / "dark.wt5")

ref.mean[:] -= blank.mean[:]

substrate = ref.mean[:][:,1300:1400].mean(axis=1)

ref.print_tree()
ref = ref.split("yindex", [1100, 1300])[1]
ref.create_channel(name="subtr", values=(ref.mean[:] - substrate[:, None]) / substrate[:, None], signed=True)
ref.convert("eV")

image = wt.data.from_Solis(here / "20x image 630 nm bandpass.asc").split("yindex", [985, 1185])[1].split("xindex", [1221, 1421])[1]
image.smooth(2)

for data in [image, ref]:
    y0 = data.yindex[:].mean()
    data.create_variable("y", values=(data.yindex[:] - y0) * um_per_pixel, units="um")

x0 = image.xindex[:].mean()
image.create_variable("x", values=(image.xindex[:]-x0) * um_per_pixel, units="um")

image.transform("x", "y")
ref.transform("wm", "y")

fig, gs = wt.artists.create_figure(cols=[1, 1, "cbar"])

ax0 = plt.subplot(gs[0])
ax0.imshow(image, cmap="Greys_r", vmin=image.signal.min())
# ax0.vlines([1321], 500, 1500, color="k", ls="--", lw=2)
# ax0.set_ylim(image.yindex.min(), image.yindex.max())

ax1 = plt.subplot(gs[1])
plt.yticks(visible=False)
ax1.pcolormesh(ref, channel="subtr")

plt.show()
