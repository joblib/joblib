Tasks at hand on joblib, in increasing order of difficulty.

* Add a changelog!

* In parallel: need to deal with return arguments that don't pickle.

* Improve test coverage and documentation 

* Store a repr of the arguments for each call in the corresponding
  cachedir

* Try to use Mike McKerns's Dill pickling module in Parallel:
  Implementation idea: 
    * Create a new function that is wrapped and takes Dillo pickles as 
      inputs as output, feed this one to multiprocessing
    * pickle everything using Dill in the Parallel object.
      http://dev.danse.us/trac/pathos/browser/dill

* Make a sensible error message when wrong keyword arguments are given,
  currently we have::

    from joblib import Memory
    mem = Memory(cachedir='cache')

    def f(a=0, b=2):
	return a, b

    g = mem.cache(f)
    g(c=2) 

    /home/varoquau/dev/joblib/joblib/func_inspect.pyc in filter_args(func,
		ignore_lst, *args, **kwargs), line 168

	    TypeError: Ignore list for diffusion_reorder() contains and
			unexpected keyword argument 'cachedir'

* add a 'depends' keyword argument to memory.cache, to be able to
  specify that a function depends on other functions, and thus that the
  cache should be cleared.

* add a 'argument_hash' keyword argument to Memory.cache, to be able to
  replace the hashing logic of memory for the input arguments. It should
  accept as an input the dictionary of arguments, as returned in
  func_inspect, and return a string.

* add a sqlite db for provenance tracking. Store computation time and usage 
  timestamps, to be able to do 'garbage-collection-like' cleaning of
  unused results, based on a cost function balancing computation cost and
  frequency of use.


