import subprocess

from distributed import LocalCluster
from dask_jobqueue import SLURMCluster

# WARNING: all of these constants are specific to the INRIA's margaret cluster

ALL_NODES = [f"node{i:02d}" for i in range(1, 33)]

# system limit above with jobs are marked as pendinng, I don't know why.
MAX_JOBS_PER_NODE = 4

# each call dask-worker will generate 4 worker processes. This achieves a good
# tradeoff between node usage (given the limit above) and fine-grain
# benchmarking.
WORKER_PER_JOBS = 4


def get_sbatch_args(n_workers):
    """generates slurm args to use an optimal number of nodes given the
    number of requested workers
    """
    assert n_workers // WORKER_PER_JOBS

    num_nodes = (n_workers - 1) // (MAX_JOBS_PER_NODE * WORKER_PER_JOBS) + 1

    cmd = "sinfo -N | grep node | grep idle | cut -d' ' -f 1"
    nodes_to_use = subprocess.check_output(cmd, shell=True)
    nodes_to_use = nodes_to_use.decode().strip().split("\n")

    assert len(nodes_to_use) > num_nodes
    nodes_to_use = nodes_to_use[-num_nodes:]

    excluded_nodes = [n for n in ALL_NODES if n not in nodes_to_use]
    excluded_nodes_str = ",".join(excluded_nodes)
    print(excluded_nodes_str)
    print(f"will use nodes: {nodes_to_use}")
    return f"--oversubscribe -x {excluded_nodes_str}"


def create_dask_cluster(
    use_slurm: bool, n_workers: int, threads_per_worker: int
):
    if use_slurm:
        cluster = SLURMCluster(
            workers=0,  # number of (initial slurm jobs)
            memory="16GB",
            extra=['--nthreads 1 --nprocs=4'],  # arguments to dask-worker CLI
            job_extra=[get_sbatch_args(n_workers)],
        )
        num_jobs = (n_workers - 1) // WORKER_PER_JOBS + 1
        cluster.scale(num_jobs)
    else:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=True,
        )
    return cluster
