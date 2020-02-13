import os
import time
import multiprocessing
from joblib import Parallel, delayed
import joblib



def get_pid():
    time.sleep(1)
    return multiprocessing.current_process().pid

def runner(i, gpu_lookup):
    pid = get_pid()
    gpu_id = gpu_lookup[pid]
    print('task: {0} running on gpu: {1} {2}'.format(i, gpu_id, joblib.worker_id()))

print(joblib.__path__)
N_GPU = 4
pids = Parallel(n_jobs=N_GPU)(delayed(get_pid)() for i in range(N_GPU))
assert len(set(pids)) == N_GPU
gpu_lookup = {pid: i for i, pid in enumerate(pids)}

Parallel(n_jobs=N_GPU)(delayed(runner)(i, gpu_lookup)
                       for i in range(12))

#p = Parallel()
#print(dir(p))
#print(dir(p._backend))
#print(dir(p._backend._workers))
