import os
from os.path import join
import sys
import time
import numpy as np
from multiprocessing import Pool

# Global Constants
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
    # Improvement: Define view outside loop to reduce overhead
    u_interior = u[1:-1, 1:-1]

    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        
        # Convergence check
        delta = np.abs(u_interior[interior_mask] - u_new_interior).max()
        
        # Update interior
        u_interior[interior_mask] = u_new_interior
        
        if delta < atol:
            break
    return u

def worker_task(bid):
    u0, interior_mask = load_data(bid)
    _ = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    return True 

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python Task6.py <num_buildings> <num_workers>")
        sys.exit(1)

    N = int(sys.argv[1])
    num_workers = int(sys.argv[2])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()[:N]

    start_time = time.perf_counter()
    
    with Pool(processes=num_workers) as pool:
        # DYNAMIC SCHEDULING: 
        # By setting chunksize=1, workers pull one task at a time.
        # This provides automatic load balancing.
        pool.map(worker_task, building_ids, chunksize=1)

    end_time = time.perf_counter()
    
    print(f"{num_workers}, {end_time - start_time:.4f}")