"""
====================================
How to use tqdm with joblib.Parallel
====================================

This example illustrates how to use ``tqdm`` progress bar along with
:class:`joblib.Parallel`.

We will see two main ways to do so. The first one,
very simple, could be used when using a generator as output of a
:class:`~joblib.Parallel` call. When the ouput is not a generator (*e.g.*
using ``return_as='list'``) we will have to define a subclass of
:class:`~joblib.Parallel`.

"""

##############################################################################
# Using generator
##############################################################################

##############################################################################
# We will first define a task that will sleep for a time given as parameter
# and return this same time. We also define an array of times that will be
# used.

import time

times = [7,2,3,5,6,4,1]

def task(t):
	time.sleep(0.1*t)
	return t

##############################################################################
# A ``tqdm`` progress bar takes as input an iterable.
# A :class:`~joblib.Parallel` call also returns an iterable over the outputs
# of each task. By default, this iterable is a ``list``. As we will see in
# the following example, it is not convenient as we have to wait for all tasks
# to be completed before obtaining the output ``list``.

from tqdm import tqdm
from joblib import Parallel, delayed

p = Parallel(n_jobs=2)
out = p(delayed(task)(t) for t in times)
print(*tqdm(out, total=len(times)))

##############################################################################
# As a result, the progress bar only appears after all tasks are completed and
# it immediately reaches 100%.
#
# A solution is to use ``return_as='generator'`` to obtain a generator that
# iterates over the outputs of each task.

p = Parallel(n_jobs=2, return_as='generator')
out = p(delayed(task)(t) for t in times)
print(*tqdm(out, total=len(times)))

##############################################################################
# The progess bar seems to work now. But we can still observe that it jumps
# from 1/7 to 4/7. This is because when using ``return_as='generator'``, the
# outputs are returned in the same order the tasks where given. So the first
# value is the output of the tasks taking 0.7s.
# (remember ``times = [7,2,3,5,6,4,1]``). When the first task is completed,
# the second and third tasks should also have completed as 2+3 < 7. That's
# why we jumped from 1/7 to 4/7.
# 
# If you don't care about the order of the outputs and want to get a better
# reporting, it is possible to use ``return_as='generator_unordered'``. The
# generator is then unordered, meaning that if the second task finishes before
# the first one, it will appear first in the generator allowing the progress
# bar to advance without waiting the first task to finish.

p = Parallel(n_jobs=2, return_as='generator_unordered')
out = p(delayed(task)(t) for t in times)
print(*tqdm(out, total=len(times)))

##############################################################################
# The progress bar now advances one iteration at a time. Observe also, how
# the order of the outputs changed in the last line.

##############################################################################
# Using a subclass of :class:`joblib.Parallel`
##############################################################################

##############################################################################
# It's not always possible to use a generator as output (`e.g.` when using
# ``backend='multiprocessing'``). In that case, it's still possible to use
# ``tqdm`` bu we must rely on a new subclass of :class:`~joblib.Parallel`.
# The :class:`~joblib.Parallel` has a method
# :meth:`~joblib.Parallel.print_progress` which is called each time a task is
# completed. This function originaly prints more or less information depending
# on the ``verbose`` parameter. Here we can simply override it in order to
# update our progress bar instead.

class ParallelTqdm(Parallel):
	def __call__(self, iterable, n_tasks):
		self.tqdm = tqdm(total=n_tasks)
		return super().__call__(iterable)

	def print_progress(self):
		self.tqdm.update()
		if self.n_completed_tasks == self.tqdm.total:
			self.tqdm.close()

p = ParallelTqdm(n_jobs=2)
out = p((delayed(task)(t) for t in times), len(times))
print(*out)
