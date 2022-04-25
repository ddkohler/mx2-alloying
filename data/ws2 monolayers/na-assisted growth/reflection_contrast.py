import WrightTools as wt
import pathlib
import matplotlib.pyplot as plt


HERE = pathlib.Path(__file__).resolve().parent


def main():
    """ reflection contrast
    """
    root = wt.open(HERE / "root.wt5")
    data = root.spectrum
    data.convert("eV")
    data.create_channel("rc_spot2", values=data.spot2[:] / data.signal[:] - 1, signed=True)
    data.create_channel("rc_spot1", values=data.spot1[:] / data.signal[:] - 1, signed=True)
    data.smooth(2)

    wt.artists.interact2D(data, channel="rc_spot2")
    plt.show()


if __name__ == "__main__":
    main()
