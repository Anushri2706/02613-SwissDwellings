from os.path import join
import sys

import numpy as np
from numba import njit
from time import time


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u

@njit(cache=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-6):
    # using values from previous iteration to compute the next iteration
    u_old = u.copy()
    u_new = u.copy()

    rows, cols = interior_mask.shape  # 512 x 512

    for iteration in range(max_iter):
        delta = 0.0

        # ensuring the use of spatial locality by iterating row-wise and column-wise
        for i in range(rows):
            for j in range(cols):
                if interior_mask[i, j]:
                    # interior_mask[i, j] corresponds to u[i+1, j+1]
                    new_value = 0.25 * (
                        u_old[i + 1, j] +       # left
                        u_old[i + 1, j + 2] +   # right
                        u_old[i, j + 1] +       # up
                        u_old[i + 2, j + 1]     # down
                    )

                    diff = abs(u_old[i + 1, j + 1] - new_value)
                    if diff > delta:
                        delta = diff

                    u_new[i + 1, j + 1] = new_value

        # did solution stop changing?
        if delta < atol:
            return u_new

        # avoid copying the full array every iteration
        tmp = u_old
        u_old = u_new
        u_new = tmp

    return u_old


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    # run the function once for accurate timing
    load_run = jacobi_numba(all_u0[0], all_interior_mask[0], 1, ABS_TOL)

    all_u = np.empty_like(all_u0)
    start = time()
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
    end = time()
    print(f"Total simulation time for {N} buildings: {end - start:.2f} seconds")
    estimated_seconds = (end - start) / N * 4571
    print(f"Estimated full dataset runtime: {estimated_seconds:.2f} seconds")
    print(f"Estimated full dataset runtime: {estimated_seconds / 60:.2f} minutes")
    print(f"Estimated full dataset runtime: {estimated_seconds / 3600:.2f} hours")
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))