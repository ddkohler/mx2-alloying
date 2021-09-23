import WrightTools as wt
import numpy as np
import pathlib

here = pathlib.Path(__file__).resolve().parent

# slit parameters
mpix = 1333  # middle pixel
# deltax = 4
# lpix_im = mpix-deltax
# rpix_im = mpix+deltax
um_per_pixel = 0.3  # 20x

root = wt.Collection(name='root')

# --- lamp ---
temp = wt.data.from_Solis(here / '20x image.asc').split("yindex", [850, 1300])[1].split("xindex", [1050, 1500])[1]
temp.create_variable(
    "ydist", values=(temp.yindex[:] - temp.yindex[:].mean()) * um_per_pixel, units="um"
)
temp.create_variable(
    "xdist", values=(temp.xindex[:] - mpix) * um_per_pixel, units="um"
)
temp.transform("xdist", "ydist")
temp.copy(parent=root, name="lamp")

# --- spectrum ---
temp = wt.data.from_Solis(here / '20x spectrum.asc').split("yindex", [850, 1300])[1]
# ddk: curious about this assignment...
temp.create_variable(
    "energy", values=np.linspace(440.1, 800.9, temp.wm[:].size).reshape(temp.wm.shape), units="nm"
)
temp.energy.convert("eV")
temp.create_variable(
    "ydist", values=(temp.yindex[:] - temp.yindex[:].mean()) * um_per_pixel, units="um"
)
temp.smooth((10,0))

substrate_low = temp.split("yindex", [900])[0].signal[:].mean(axis=1)[:, None]
substrate_high = temp.split("yindex", [1250])[1].signal[:].mean(axis=1)[:, None]
# interpolate between top and bottom substrate spectra 
z = temp.yindex[:].copy()
s = (z - 900) / 350
s[s>1] = 1
s[s<0] = 0
substrate = (1-s) * substrate_low + s * substrate_high

temp.create_channel(
    "contrast", values=(temp.signal[:] - substrate) / substrate, signed=True
)

temp.copy(parent=root, name='spectrum')

root.print_tree()

root.save(here / 'reflection_20x.wt5')
