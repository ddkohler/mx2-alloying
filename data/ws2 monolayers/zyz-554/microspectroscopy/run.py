import WrightTools as wt
import matplotlib.pyplot as plt
import pathlib

here = pathlib.Path(__file__).resolve().parent

if False:
    d = wt.data.from_Solis(here / "20x ref spectrum series vtrans slit 100 um.asc", name="ref")
    d.create_channel("stdev", values=d.signal[:].std(axis=0)[None, :, :])
    d.create_channel("mean", values=d.signal[:].mean(axis=0)[None, : ,:])
    d.prune(keep_channels=(1,2))
    d.transform("wm", "yindex")
    d.save(here / "ref.wt5")


for filename, name in [
    ["20x ref spectrum series vtrans slit 100 um.asc", "ref"],
    ["dark counts.asc", "dark"],
    ["baseline exposure.asc", "baseline"],
]:
    d = wt.data.from_Solis(here / filename)
    d.create_channel("stdev", values=d.signal[:].std(axis=0)[None, :, :])
    d.create_channel("mean", values=d.signal[:].mean(axis=0)[None, : ,:])
    d.prune(keep_channels=(1,2))
    d = d.chop("wm", "yindex", at={"frame":[0, None]})[0]
    d = d.copy(name=name)
    d.transform("wm", "yindex")
    out = wt.artists.interact2D(d, channel="mean")
    plt.show()
    d.save(here / f"{name}.wt5")

