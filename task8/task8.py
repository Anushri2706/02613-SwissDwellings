from os.path import join
import sys
import time

from numba import cuda
import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    """Perform a single Jacobi iteration. Each thread updates one cell."""
    i, j = cuda.grid(2)

    # Bounds check against the interior mask (512x512)
    if i >= interior_mask.shape[0] or j >= interior_mask.shape[1]:
        return

    # Interior cell (i, j) corresponds to u[i+1, j+1] (u is 514x514, padded)
    if interior_mask[i, j]:
        u_new[i + 1, j + 1] = 0.25 * (
            u[i,     j + 1] +   # up
            u[i + 2, j + 1] +   # down
            u[i + 1, j    ] +   # left
            u[i + 1, j + 2]     # right
        )
    else:
        u_new[i + 1, j + 1] = u[i + 1, j + 1]


def jacobi_cuda(u, interior_mask, max_iter):
    """Helper that uploads data, repeatedly launches the kernel, and returns the result."""
    # Upload to device once
    d_u     = cuda.to_device(u)
    d_u_new = cuda.to_device(u.copy())
    d_mask  = cuda.to_device(interior_mask)

    # Launch configuration: 16x16 threads per block
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (interior_mask.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (interior_mask.shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # Each kernel launch is an implicit global barrier; ping-pong the buffers
    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](d_u, d_u_new, d_mask)
        d_u, d_u_new = d_u_new, d_u

    return d_u.copy_to_host()


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

    # Run jacobi iterations for each floor plan (fixed iteration count, no early stop)
    MAX_ITER = 20_000

    # Warm up: trigger JIT compilation so the first floorplan timing isn't skewed
    _ = jacobi_cuda(all_u0[0], all_interior_mask[0], 1)
    cuda.synchronize()

    all_u = np.empty_like(all_u0)
    per_floorplan_times = []
    t_total_start = time.perf_counter()
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        t0 = time.perf_counter()
        all_u[i] = jacobi_cuda(u0, interior_mask, MAX_ITER)
        cuda.synchronize()  # ensure kernel + copy-back actually finished
        t1 = time.perf_counter()
        per_floorplan_times.append(t1 - t0)
    t_total = time.perf_counter() - t_total_start

    # Timing summary (printed to stderr so it doesn't pollute the CSV)
    print(f"\n--- Timing (Numba CUDA) ---", file=sys.stderr)
    print(f"N floorplans:       {N}", file=sys.stderr)
    print(f"Total time:         {t_total:.3f} s", file=sys.stderr)
    print(f"Mean per floorplan: {np.mean(per_floorplan_times):.3f} s", file=sys.stderr)
    print(f"Min / Max:          {np.min(per_floorplan_times):.3f} / {np.max(per_floorplan_times):.3f} s", file=sys.stderr)
    TOTAL_FLOORPLANS = 4571
    est_full = np.mean(per_floorplan_times) * TOTAL_FLOORPLANS
    print(f"Est. full dataset ({TOTAL_FLOORPLANS}): {est_full:.1f} s = {est_full/60:.1f} min = {est_full/3600:.2f} h", file=sys.stderr)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))