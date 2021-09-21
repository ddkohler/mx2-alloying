import WrightTools as wt
import numpy as np
import pathlib

here = pathlib.Path(__file__).resolve().parent

# slit parameters
mpix = 1333 #middle pixel
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
    "energy", values=np.linspace(2.57, 1.45, temp.wm[:].size).reshape(temp.wm.shape), units="eV"
)
temp.create_variable(
    "ydist", values=(temp.yindex[:] - temp.yindex[:].mean()) * um_per_pixel, units="um"
)
temp.copy(parent=root, name='spectrum')

root.print_tree()

root.save(here / 'reflection_20x.wt5')
