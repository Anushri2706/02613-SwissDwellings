from os.path import join
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
OUT_DIR = "plots"


def load_inputs(bid):
    domain = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return domain, interior


def main():
    os.makedirs(OUT_DIR, exist_ok = True)

    with open(join(LOAD_DIR, "building_ids.txt")) as file:
        building_ids = file.read().splitlines()

    n = int(sys.argv[1])

    for building_id in building_ids[:n]:
        domain, interior = load_inputs(building_id)

        fig, axes = plt.subplots(1, 2, figsize = (10, 5))

        im0 = axes[0].imshow(domain, cmap = "viridis")
        axes[0].set_title(f"{building_id} domain")
        axes[0].axis("off")
        fig.colorbar(im0, ax = axes[0], fraction = 0.046, pad = 0.04)

        im1 = axes[1].imshow(interior, cmap = "gray")
        axes[1].set_title(f"{building_id} interior mask")
        axes[1].axis("off")
        fig.colorbar(im1, ax = axes[1], fraction= 0.046, pad = 0.04)

        fig.tight_layout()
        fig.savefig(join(OUT_DIR, f"{building_id}_inputs.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()