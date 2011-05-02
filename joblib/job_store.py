# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

"""The default job stores

Design
------

The job store is the heart of joblib; classes like Memory primarily
act as a front-end. It is responsible for:

 - Mapping functions and inputs to "jobs"
 
 - Caching jobs, including implementation, being in charge of caching
   policies, and also determining which jobs are considered "equal"
 
 - Locking; it should be reentrant (as discussed below) and is
   responsible for being failsafe. There API also provides for
   pessimistic locking, but stores do not need to support this.

It is *not* responsible for carrying out computations, which is
left to the client.

The default implementation used by joblib is DirectoryJobStore, based on
pickling output to file in a directory tree. To configure how jobs are
stored, one can subclass either DirectoryJobStore or BaseJobStore.

Job stores should (in general) be fully reentrant, as well as
picklable when it makes sense. The ``get_job_handle`` call should
(without waiting) return a job handle object that deals with further
operations on a single job, which is *not* reentrant (although there
can be several simultaneous handles to the same job). In general, any
difficult synchronization happens within the job handles.

Take care when using job handles: get_output and persist_... can
hold on to objects as it sees fits. In general, close them as soon as
possible.

"""

# System imports
import os
from os.path import join as pjoin
import sys
import shutil
import traceback
import warnings
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    # json is in the standard library for Python >= 2.6
    import json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        # Not the end of the world: we'll do without this functionality
        json = None

# Relative imports
from .hashing import hash
from .func_inspect import get_func_name, get_func_code
from . import numpy_pickle

# Config
FIRST_LINE_TEXT = "# first line:"

# Enums.
class Enum(object):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return '<%s>' % self.name
    def __eq__(self, other):
        return self.name == other.name
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash(self.name)
    
COMPUTED = Enum('COMPUTED')
MUST_COMPUTE = Enum('MUST_COMPUTE')
WAIT = Enum('WAIT')

# Exceptions/warnings

class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """

class IllegalOperationError(Exception):
    pass


# Utils
class PrintLogger(object):
    def info(self, msg, *args):
        print msg % args
    error = debug = warn = info
print_logger = PrintLogger()

def extract_first_line(func_code):
    """ Extract the first line information from the function code
        text if available.
    """
    if func_code.startswith(FIRST_LINE_TEXT):
        func_code = func_code.split('\n')
        first_line = int(func_code[0][len(FIRST_LINE_TEXT):])
        func_code = '\n'.join(func_code[1:])
    else:
        first_line = -1
    return func_code, first_line


# Body

class BaseJobStore(object):
    def __init__(self, logger=print_logger, mmap_mode=None):
        self.logger = logger
        self.mmap_mode = mmap_mode

    def get_job(self, func, args_dict):
        input_hash = self._hash_input(args_dict)
        return self._get_job_from_hash(func, input_hash)

    # Must be overridden

    def _get_job_from_hash(self, func, input_hash):
        raise NotImplementedError()
        
    def clear(self, func, warn=True):
        """ Delete all computed results for the given function.
        """
        raise NotImplementedError()

    # Overridable

    def _hash_input(self, args_dict):
        """ Hashes arguments.
        """
        coerce_mmap = (self.mmap_mode is not None)
        input_hash = hash(args_dict,
                          coerce_mmap=coerce_mmap)
        return input_hash

class DirectoryJobStore(BaseJobStore):
    def __init__(self, path, logger=print_logger, save_npy=True, mmap_mode=None,
                 verbose=1):
        BaseJobStore.__init__(self, logger=logger, mmap_mode=mmap_mode)
        self.store_path = os.path.abspath(path)
        self.save_npy = save_npy
        self.verbose = verbose

    def _get_job_from_hash(self, func, input_hash):
        job_path = pjoin(self.store_path, self._get_func_name(func), input_hash)
        return DirectoryJob(job_path, self, func,
                            self.logger,
                            self.save_npy,
                            self.mmap_mode)

    def clear(self, func, warn=True):
        """ Delete all computed results for the given function.
        """
        func_dir = pjoin(self.store_path, self._get_func_name(func))
        if self.verbose and warn:
            self.logger.warn("Clearing cache %s" % func_dir)
        if os.path.exists(func_dir):
            shutil.rmtree(func_dir, ignore_errors=True)
        self._store_function_source_code(func)

    # Private
    def _get_func_dir(self, func):
        return pjoin(self.store_path, self._get_func_name(func))
    
    def _get_func_name(self, func):
        fplst, name = get_func_name(func)
        fplst.append(name)
        return pjoin(*fplst)

    def _store_function_source_code(self, func):
        func_code, _, first_line = get_func_code(func)
        func_code = '%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        targetpath = pjoin(self.store_path, self._get_func_name(func))
        targetfile = pjoin(targetpath, 'func_code.py')
        ensure_dir(targetpath)
        file(targetfile, 'w').write(func_code)
        
    def _check_previous_func_code(self, func, stacklevel=2):
        """ 
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
        # Here, we go through some effort to be robust to dynamically
        # changing code and collision. We cannot inspect.getsource
        # because it is not reliable when using IPython's magic "%run".
        func_code, source_file, first_line = get_func_code(func)
        func_dir = self._get_func_dir(func)
        func_code_file = os.path.join(func_dir, 'func_code.py')
        try:
            if not os.path.exists(func_code_file): 
                raise IOError
            old_func_code, old_first_line = \
                            extract_first_line(file(func_code_file).read())
        except IOError:
                self._store_function_source_code(func)
                return False
        if old_func_code == func_code:
            return True

        # We have differing code, is this because we are refering to
        # differing functions, or because the function we are refering as 
        # changed?

        if old_first_line == first_line == -1:
            _, func_name = get_func_name(func, resolv_alias=False,
                                         win_characters=False)
            if not first_line == -1:
                func_description = '%s (%s:%i)' % (func_name, 
                                                source_file, first_line)
            else:
                func_description = func_name
            warnings.warn(JobLibCollisionWarning(
                "Cannot detect name collisions for function '%s'"
                        % func_description), stacklevel=stacklevel)

        # Fetch the code at the old location and compare it. If it is the
        # same than the code store, we have a collision: the code in the
        # file has not changed, but the name we have is pointing to a new
        # code block.
        if (not old_first_line == first_line 
                                    and source_file is not None
                                    and os.path.exists(source_file)):
            _, func_name = get_func_name(func, resolv_alias=False)
            num_lines = len(func_code.split('\n'))
            on_disk_func_code = file(source_file).readlines()[
                        old_first_line-1:old_first_line-1+num_lines-1]
            on_disk_func_code = ''.join(on_disk_func_code)
            if on_disk_func_code.rstrip() == old_func_code.rstrip():
                warnings.warn(JobLibCollisionWarning(
                'Possible name collisions between functions '
                "'%s' (%s:%i) and '%s' (%s:%i)" %
                (func_name, source_file, old_first_line, 
                 func_name, source_file, first_line)),
                 stacklevel=stacklevel)

        # The function has changed, wipe the cache directory.
        # XXX: Should be using warnings, and giving stacklevel
        self.clear(func, warn=True)
        return False

    def repr_for_func(self, func):
        # TODO: Ugh. Refactor this in a better way.
        return '<cachedir:%s>' % self._get_func_dir(func)

def _noop(): pass

class DirectoryJob(object):
    """
    *always* call close on one of these
    """
    def __init__(self, job_path, store, func, logger, save_npy, mmap_mode):
        self.job_path = os.path.realpath(job_path)
        self.store = store
        self.func = func
        self.logger = logger
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        self._work_path = None

    def __del__(self):
        if self.job_path is not None:
            warnings.warn('Did not close DirectoryJob, fix your code!')

    def persist_input(self, args_tuple, kwargs_dict, filtered_args_dict):
        if self._work_path is None:
            raise IllegalOperationError("call load_or_lock first")
        if json is not None and filtered_args_dict is not None:
            input_repr = dict((k, repr(v)) for k, v in filtered_args_dict.iteritems())
            with file(pjoin(self._work_path, 'input_args.json'), 'w') as f:
                json.dump(input_repr, f)

    def persist_output(self, output):
        if self._work_path is None:
            raise IllegalOperationError("call load_or_lock first")
        # TODO: Use a temporary work path and atomically move it instead
        if self._work_path is None:
            raise IllegalOperationError("call load_or_lock first")
        filename = pjoin(self._work_path, 'output.pkl')
        if 'numpy' in sys.modules and self.save_npy:
            numpy_pickle.dump(output, filename) 
        else:
            with file(filename, 'w') as f:
                pickle.dump(output, f, protocol=2)

    def is_computed(self):
        """ Whether the job is computed or not. Use this method only
        if you know what you are doing, since it is not race-safe:

        If the return value is ``True``, then a subsequent call to
        ``load_or_lock`` is guaranteed to return ``COMPUTED``,
        up to unpickling errors.  However, if it returns ``False``,
        it could of course have changed by the time the caller gets
        the result and can act on it.
        """
        return os.path.exists(self.job_path)

    def clear(self):
        if os.path.exists(self.job_path):
            shutil.rmtree(self.job_path, ignore_errors=True)

    def load_or_lock(self, blocking=True, pre_load_hook=_noop,
                     post_load_hook=_noop):
        """ Fetch result of or offer to compute the job.

        The API supports both transactional safety and pessimistic
        locking, but this implementation provides neither.

        Returns
        -------

        (status, output)

        Status is either MUST_COMPUTE, WAIT, or COMPUTED. If
        it returns MUST_COMPUTE then you *must* call commit() or
        rollback() (although rollback() is called by close()).
        If 'blocking' is True, then WAIT can not be
        returned.

        When ``status == COMPUTED``, the output of the job is
        present in ``output``. Otherwise, ``output`` is ``None``.
        """
        self.store._check_previous_func_code(self.func, stacklevel=3)
        output = None
        if self.is_computed():
            try:
                # TODO pre_load_hook and post_load_hook are ugly
                # hacks that should go once logging framework is
                # fixed
                pre_load_hook()
                output = self._load_output()
                post_load_hook()
            except BaseException:
                # Corrupt output; recompute
                self.logger.warn(
                        'Exception while loading results for %s\n %s' % (
                        self.job_path, traceback.format_exc()))
                self.clear()
                status = MUST_COMPUTE
            else:
                status = COMPUTED
        else:
            status = MUST_COMPUTE
        if status == MUST_COMPUTE:
            self._work_path = self.job_path
            ensure_dir(self._work_path)            
        return (status, output)

    def commit(self):
        self._work_path = None

    def rollback(self):
        self._work_path = None

    def close(self):
        self.rollback()
        self.job_path = None
        self._output = None

    def _load_output(self):
        filename = pjoin(self.job_path, 'output.pkl')
        if self.save_npy:
            return numpy_pickle.load(filename, 
                                     mmap_mode=self.mmap_mode)
        else:
            with file(filename, 'r') as f:
                return pickle.load(f)        

def ensure_dir(path):
    """ Ensure that a directory exists with graceful race handling

    An exception is not raised if two processes attempt to create
    the directory at the same time; but still raised in other
    circumstances.
    """
    try:
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            pass # race condition
        else:
            raise # something else
