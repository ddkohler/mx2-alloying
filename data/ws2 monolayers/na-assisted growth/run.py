import WrightTools as wt
import pathlib
# import matplotlib.pyplot as plt


HERE = pathlib.Path(__file__).resolve().parent
print(HERE)


def main():
    """ raw images
    """
    root = wt.open(HERE / "root.wt5")
    root.print_tree()
    for data in root.values():
        wt.artists.quick2D(data)
        wt.artists.savefig(HERE / (data.natural_name + ".png"))


if __name__ == "__main__":
    main()
