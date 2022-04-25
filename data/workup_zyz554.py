import numpy as np
import matplotlib.pyplot as plt
import WrightTools as wt
import os
import pathlib
import time
from scipy.ndimage import median_filter


all_import = True
here = pathlib.Path(__file__).resolve().parent
data_dir = here / "ws2 monolayers" / "zyz-554"
root = wt.Collection(name="root")
verbose = False
talkback = True

tstart = time.time()
if talkback:
    print("starting import zyz-554")


# --- PL and Raman --------------------------------------------------------------------------------


# --- --- PL
if all_import:
    PL = root.create_collection(name="pl")
    p = data_dir / "labram"
    for name, filename in [
        ["face", "WS2_linescan_face-to-face_PL.txt"],
        ["corner", "WS2_linescan_vertex-to-vertex_PL.txt"]
    ]:

        d = wt.data.from_LabRAM(p / filename, parent=PL, name=name)
    if talkback:
        print("PL done", time.time() - tstart)

# --- --- Raman
if all_import:
    # raw
    raman = root.create_collection(name="raman")
    p = data_dir / "labram"
    for name, filename in [
        ["face", "WS2_linescan_face-to-face.txt"],
        ["corner", "WS2_linescan_vertex-to-vertex.txt"]
    ]:
        d = wt.data.from_LabRAM(p / filename, name=name, parent=raman)

    if talkback:
        print("Raman done", time.time() - tstart)


# --- reflection ----------------------------------------------------------------------------------


if all_import:
    reflection = root.create_collection(name="reflection")
    p = data_dir / "microspectroscopy"
    um_per_pixel = 0.3  # 20x

    for filename, name in [
        ["20x ref spectrum series vtrans slit 100 um.asc", "refl"],
        ["dark counts.asc", "blank"],
        # ["baseline exposure.asc", "bias"],
    ]:
        d = wt.data.from_Solis(p / filename)
        d.create_channel("stdev", values=d.signal[:].std(axis=0)[None, :, :])
        d.create_channel("mean", values=d.signal[:].mean(axis=0)[None, : ,:])
        d.prune(keep_channels=(1,2))
        d = d.chop("wm", "yindex", at={"frame":[0, None]})[0]
        d.copy(name=name, parent=reflection)
        d.transform("wm", "yindex")

    refl = reflection.refl
    refl.mean[:] -= reflection.blank.mean[:]
    substrate = refl.mean[:][:,1300:1400].mean(axis=1)
    refl.create_channel(name="subtr", values=(refl.mean[:] - substrate[:, None]) / substrate[:, None], signed=True)

    image = wt.data.from_Solis(p / "20x image 630 nm bandpass.asc").split("yindex", [985, 1185])[1].split("xindex", [1221, 1421])[1]
    image = image.copy(parent=reflection, name="image")
    image.smooth(2)

    x0 = 1321
    for data, y0 in [[image, reflection.blank.yindex.max() // 2], [refl, 1200]]:
        data.create_variable("y", values=(data.yindex[:] - y0) * um_per_pixel, units="um")

    image.create_variable("x", values=(image.xindex[:]-x0) * um_per_pixel, units="um")

    if talkback:
        print("Reflection done", time.time() - tstart)


# --- Finish --------------------------------------------------------------------------------------


if all_import:
    root.print_tree(depth=2)
    root.save(here / "zyz-554.wt5", overwrite=True)

    if talkback:
        print("DONE! Total time", time.time() - tstart)
