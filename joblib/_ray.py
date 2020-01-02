from __future__ import print_function, division, absolute_import
import ray
from .parallel import AutoBatchingMixin, ParallelBackendBase

class Future:
    """ Future class as patch to combine joblib futures 
    with Ray futures and support Async API.
    Args:
        rayfuture(future): the future from ray's func.remote call.
    Attributes:
        rayfuture(future): the future from ray's func.remote call.
    """
    def __init__(self,rayfuture):
        self.rayfuture = rayfuture
    
    def get(self,timeout=None):
        """ get function to support future.get() of joblib.
        Args:
            timeout(float): timeout in seconds to wait for the future.
        Returns:
            result: the result returned from the future. 
        Raises:
            TimeoutError: if the task pointed by the future times out.
        """
        if timeout:
            done_futures,remaining_futures = ray.wait([self.rayfuture],1,timeout)
            if not done_futures:
                raise TimeoutError()
        result = ray.get(self.rayfuture)
        return result

@ray.remote
def executer(func):
    """ decorates the functions to execute 
        Args:
            func: the function to decorate and execute in Ray.
        Returns:
            The output of the function.
    """
    return func()


class RayBackend(ParallelBackendBase, AutoBatchingMixin):
    supports_timeout = True
    MIN_IDEAL_BATCH_DURATION = 0.2
    MAX_IDEAL_BATCH_DURATION = 1.0
    #supports_sharedmem = True
    
    def __init__(self,**ray_kwargs):
        """Ray backend uses ray, a system for scalable distributed computing.
        more info are available here: https://ray.readthedocs.io/
        
        Args:
            ray_kwargs: the parameters to pass to ray.init().
        Attributes:
            ray_kwargs: the parameters to pass to ray.init(). Used to 
            retrieve the configurations after running abort_everything.
            task_future(list of futures): to keep track of running futures
        """
        
        self.ray_kwargs = ray_kwargs
        self.task_futures = set()
        if 'num_cpus' not in self.ray_kwargs:
            self.ray_kwargs['num_cpus'] = 1
        ray.init(ignore_reinit_error=True,**self.ray_kwargs)
    
    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs/workers which are going 
        to run in parallel. Currently it ignores n_jobs and return num_cpus"""
        return self.ray_kwargs.get('num_cpus')

    def get_nested_backend(self):
        return self, -1

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        """n_jobs is basically managed by ray. 
        Make sure to define num_cpus in the init"""
        n_jobs = self.effective_n_jobs(n_jobs)
        if n_jobs == None: 
            n_jobs = 1
        return n_jobs
    

    def apply_async(self, func, callback=None):
        """Schedule a func to be run"""
        rayfuture = executer.remote(func)
        self.task_futures.add(rayfuture)

        if callback is not None:    
            callback(rayfuture)
            self.task_futures.remove(rayfuture)
        
        return Future(rayfuture)
    
    def abort_everything(self,ensure_ready=True):
        """ shuts down ray (and running jobs) and reinitializes the class """
        ray.shutdown()
        self.task_futures.clear()
        if ensure_ready:
            ray.init(**self.ray_kwargs)
            self.configure()
