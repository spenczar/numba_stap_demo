from clusters import find_clusters_hotspots_2d
from numba_stap import enable_probes
import numpy as np


def main():
    data = np.genfromtxt("input.txt")
    i = 0
    while True:
        i += 1
        print(f"iteration {i}")
        find_clusters_hotspots_2d(data, 0.005, 5)


if __name__ == "__main__":
    main()
