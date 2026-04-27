import os
from os.path import join
import sys
import time
import numpy as np
from multiprocessing import Pool

# Global Constants - Visible to all worker processes
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4

def load_data(bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol):
    u = np.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def worker_task(bid):
    """Function executed by each worker process for one floorplan."""
    u0, interior_mask = load_data(bid)
    final_u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    # Return a dummy result to minimize IPC overhead for timing
    return True 

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python Task5.py <num_buildings> <num_workers>")
        sys.exit(1)

    N = int(sys.argv[1])
    num_workers = int(sys.argv[2])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    # STATIC SCHEDULING: 
    # Calculate chunksize so each worker gets an equal share upfront
    chunksize = int(np.ceil(N / num_workers))

    start_time = time.perf_counter()
    
    with Pool(processes=num_workers) as pool:
        # map() with chunksize implements static scheduling
        pool.map(worker_task, building_ids, chunksize=chunksize)

    end_time = time.perf_counter()
    
    # Output: workers, execution_time (easy for the bash script to collect)
    print(f"{num_workers}, {end_time - start_time:.4f}")