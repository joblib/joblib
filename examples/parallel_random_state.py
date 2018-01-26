"""
===================================
Random state within joblib.Parallel
===================================

Randomness is affected by parallel execution differently by the different
backends.

In particular, when using multiple processes, the random sequence can be
the same in all processes. This example illustrates the problem and shows
how to work around it.
"""

import numpy as np
from joblib import Parallel, delayed


###############################################################################
# A utility function for the example
def print_vector(vector, backend):
    """Helper function to print the generated vector with a given backend."""
    print('\nThe different generated vectors using the {} backend are:\n {}'
          .format(backend, np.array(vector)))


###############################################################################
# Sequential behavior
###############################################################################
#
# ``stochastic_function`` will generate five random integers. When
# calling the function several times, we are expecting to obtain
# different vectors. For instance, we will call the function five times
# in a sequential manner, we can check that the generated vectors are all
# different.


def stochastic_function(max_value):
    """Randomly generate integer up to a maximum value."""
    return np.random.randint(max_value, size=5)


n_vectors = 5
random_vector = [stochastic_function(10) for _ in range(n_vectors)]
print('\nThe different generated vectors in a sequential manner are:\n {}'
      .format(np.array(random_vector)))

###############################################################################
# Parallel behavior
###############################################################################
#
# Joblib provides three different backend: loky (default), threading, and
# multiprocessing.

backend = 'loky'
random_vector = Parallel(n_jobs=2, backend=backend)(delayed(
    stochastic_function)(10) for _ in range(n_vectors))
print_vector(random_vector, backend)

###############################################################################

backend = 'threading'
random_vector = Parallel(n_jobs=2, backend=backend)(delayed(
    stochastic_function)(10) for _ in range(n_vectors))
print_vector(random_vector, backend)

###############################################################################
# Loky and the threading backends behave exactly as in the sequential case and
# do not require more care. However, this is not the case regarding the
# multiprocessing backend.

backend = 'multiprocessing'
random_vector = Parallel(n_jobs=2, backend=backend)(delayed(
    stochastic_function)(10) for _ in range(n_vectors))
print_vector(random_vector, backend)

###############################################################################
# Some of the generated vectors are exactly the same, which can be a
# problem for the application.
#
# Technically, the reason is that all forked Python processes share the
# same exact random seed. As a results, we obtain twice the same randomly
# generated vectors because we are using ``n_jobs=2``. A solution is to
# set the random state within the function which is passed to
# :class:`joblib.Parallel`.


def stochastic_function_seeded(max_value, random_state):
    rng = np.random.RandomState(random_state)
    return rng.randint(max_value, size=5)


###############################################################################
# ``stochastic_function_seeded`` accepts as argument a random seed. We can
# reset this seed by passing ``None`` at every function call. In this case, we
# see that the generated vectors are all different.

random_vector = Parallel(n_jobs=2, backend=backend)(delayed(
    stochastic_function_seeded)(10, None) for _ in range(n_vectors))
print_vector(random_vector, backend)

###############################################################################
# Fixing the random state to obtain deterministic results
###############################################################################
#
# The pattern of ``stochastic_function_seeded`` has another advantage: it
# allows to control the random_state by passing a known seed. So for instance,
# we can replicate the same generation of vectors by passing a fixed state as
# follows.

random_state = np.random.randint(np.iinfo(np.int32).max, size=n_vectors)

random_vector = Parallel(n_jobs=2, backend=backend)(delayed(
    stochastic_function_seeded)(10, rng) for rng in random_state)
print_vector(random_vector, backend)

random_vector = Parallel(n_jobs=2, backend=backend)(delayed(
    stochastic_function_seeded)(10, rng) for rng in random_state)
print_vector(random_vector, backend)
