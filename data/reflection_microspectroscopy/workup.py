import WrightTools as wt
import matplotlib as mpl
import os
import numpy as np

__here__ = os.path.abspath(os.path.dirname(__file__))
um_per_pixel = 0.06

tweak_contrast = True  # tweak the reflection intensities to match at substrate
y0 = 1055  # pixel that marks the heterostructure transition (y=0)
x0 = 1338  # slit center pixel

root = wt.Collection(name="reflection")

img_sig = wt.data.from_Solis(
    os.path.join(__here__, "target area 100x image.asc")
)
# image is cropped:  set correct y index values
bottom = int(img_sig.attrs["{left,right,bottom,top}"].split(",")[2])
y = img_sig["yindex"]  # the x reference is necessary to use setitem (*=, +=, etc.) syntax on a variable
y += bottom 

spectrum_sig = wt.data.from_Solis(
    os.path.join(__here__, "target area 100x spectrum.asc")
)
# slight vertical axis adjustment to account for grating misalignment (known issue)
y = spectrum_sig["yindex"]
y -= 12

spectrum_blank = wt.data.from_Solis(
    os.path.join(__here__, "blank 100x spectrum.asc")
)

scalar = 0.88 if tweak_contrast else 1.
spectrum_sig.create_channel(
    name="contrast",
    # ddk: adding tweak factor to account for lamp fluctuations
    # guiding principle is to minimize delta R at substrate region
    values=scalar * spectrum_sig.signal[:] / spectrum_blank.signal[:] - 1,
    signed=True
)

spectrum_sig.smooth(5, channel="contrast")

img_sig.create_variable(name="y", values=(img_sig.yindex[:]-y0) * um_per_pixel, units="um")
img_sig.create_variable(name="x", values=(img_sig.xindex[:]-x0) * um_per_pixel, units="um")
spectrum_sig.create_variable(name="y", values=(spectrum_sig.yindex[:]-y0) * um_per_pixel, units="um")

spectrum_sig = spectrum_sig.split("wm", [1.7, 2.5], units="eV")[1]
spectrum_sig = spectrum_sig.split("y", [-30, 30])[1]
img_sig = img_sig.split("y", [-30, 30])[1]
img_sig = img_sig.split("x", [-30, 30])[1]

img_sig.transform("x", "y")
spectrum_sig.transform("wm", "y")
spectrum_sig.convert("eV")

mos2, ws2 = spectrum_sig.split("y", [0]).values()
mos2s = mos2.split("wm", [1.9, 2.2], units="eV")
ws2s = ws2.split("wm", [2, 2.4], units="eV")
ares = []
bres = []
for split in [mos2s, ws2s]:
    a, b = [split[0], split[1]]
    if split == ws2s:  # try to avoid room light feature
        b = b.split("wm", [2.3], units="eV")[1]
    ares.append(a.wm[:][a.contrast[:].argmin(axis=0)][:, 0])
    bres.append(b.wm[:][b.contrast[:].argmin(axis=0)][:, 0])
ares = spectrum_sig.create_channel(name="ares", values=np.concatenate(ares)[None,:])
bres = spectrum_sig.create_channel(name="bres", values=np.concatenate(bres)[None,:])

spectrum_sig = spectrum_sig.copy(name="spectrum", parent=root)
img_sig = img_sig.copy(name="image", parent=root)

root.save(os.path.join(__here__, "reflection.wt5"))
root.print_tree()
