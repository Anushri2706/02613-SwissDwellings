from os.path import join
import sys
import time

import numpy as np
import cupy as cp


def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter):
    u = cp.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        u[1:-1, 1:-1][interior_mask] = u_new_interior
    return u


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
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000

    all_u = cp.empty_like(all_u0)
    per_floorplan_times = []
    t_total_start = time.perf_counter()
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        t0 = time.perf_counter()
        u = jacobi(u0, interior_mask, MAX_ITER)
        t1 = time.perf_counter()
        per_floorplan_times.append(t1 - t0)
        all_u[i] = u
    t_total = time.perf_counter() - t_total_start

    # Timing summary (printed to stderr so it doesn't pollute the CSV)
    print(f"\n--- Timing (Naive Cupy) ---", file=sys.stderr)
    print(f"N floorplans:       {N}", file=sys.stderr)
    print(f"Total time:         {t_total:.3f} s", file=sys.stderr)
    print(f"Mean per floorplan: {np.mean(per_floorplan_times):.3f} s", file=sys.stderr)
    print(f"Min / Max:          {np.min(per_floorplan_times):.3f} / {np.max(per_floorplan_times):.3f} s", file=sys.stderr)
    # Extrapolate to full dataset (4571 floorplans is the full set; adjust as needed)
    TOTAL_FLOORPLANS = 4571
    est_full = np.mean(per_floorplan_times) * TOTAL_FLOORPLANS
    print(f"Est. full dataset ({TOTAL_FLOORPLANS}): {est_full:.1f} s = {est_full/60:.1f} min = {est_full/3600:.2f} h", file=sys.stderr)

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))