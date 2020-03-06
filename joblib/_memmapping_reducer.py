"""
Reducer using memory mapping for numpy arrays
"""
# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Copyright: 2017, Thomas Moreau
# License: BSD 3 clause

from mmap import mmap
import errno
import os
import stat
import threading
import atexit
import sys
import tempfile
import time
import warnings
import weakref
from uuid import uuid4
from multiprocessing import util

from pickle import whichmodule, loads, dumps, HIGHEST_PROTOCOL, PicklingError

try:
    WindowsError
except NameError:
    WindowsError = type(None)

try:
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
except ImportError:
    np = None

from .numpy_pickle import dump, load, load_temporary_memmap
from .numpy_pickle_utils import _get_backing_memmap
from .backports import make_memmap
from .disk import delete_folder
from .externals.loky.backend import resource_tracker
from .externals.loky.backend.resource_tracker import _CLEANUP_FUNCS

# Some system have a ramdisk mounted by default, we can use it instead of /tmp
# as the default folder to dump big arrays to share with subprocesses.
SYSTEM_SHARED_MEM_FS = '/dev/shm'

# Minimal number of bytes available on SYSTEM_SHARED_MEM_FS to consider using
# it as the default folder to dump big arrays to share with subprocesses.
SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(2e9)

# Folder and file permissions to chmod temporary files generated by the
# memmapping pool. Only the owner of the Python process can access the
# temporary files and folder.
FOLDER_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
FILE_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR

# Set referencing the filenames of temporary memmaps created by joblib to speed
# up data communication between workers. This object need not be specific to a
# Parallel object unless we decide to support the case of calling several
# Parallel object run in different threads of a same process.
JOBLIB_MMAPS = set()

JOBLIB_MMAPS_FINALIZERS = []


def add_maybe_unlink_finalizer(memmap):
    util.debug(
        "[FINALIZER ADD] adding finalizer to {} (id {}, filename {}, pid  {} )"
        "".format(type(memmap), id(memmap), os.path.basename(memmap.filename),
                  os.getpid()))
    filename = memmap.filename

    def _maybe_unlink(wr):
        # A weakref callback is constrained to be a function with one
        # argument that is the weakref itself, while the
        # resource_tracker.maybe_unlink needs the filename and the type of
        # the resource to unlink. To workaround this limitation, we create
        # a nested function whose closure will contain the arguments of
        # resource_tracker.maybe_unlink.
        from .externals.loky.backend.resource_tracker import _resource_tracker
        util.debug(
            "[FINALIZER CALL] object mapping to {} about to be deleted,"
            " decrementing the refcount of the file (pid: {})".format(
                os.path.basename(filename), os.getpid()))
        _resource_tracker.maybe_unlink(filename, "file")

    if sys.version_info[0] == 2:
        w = weakref.ref(memmap, _maybe_unlink)
        JOBLIB_MMAPS_FINALIZERS.append(w)
    else:
        weakref.finalize(memmap, _maybe_unlink, (None,))


class _WeakArrayKeyMap:
    """A variant of weakref.WeakKeyDictionary for unhashable numpy arrays.

    This datastructure will be used with numpy arrays as obj keys, therefore we
    do not use the __get__ / __set__ methods to avoid any conflict with the
    numpy fancy indexing syntax.
    """

    def __init__(self):
        self._data = {}

    def get(self, obj):
        ref, val = self._data[id(obj)]
        if ref() is not obj:
            # In case of race condition with on_destroy: could never be
            # triggered by the joblib tests with CPython.
            raise KeyError(obj)
        return val

    def set(self, obj, value):
        key = id(obj)
        try:
            ref, _ = self._data[key]
            if ref() is not obj:
                # In case of race condition with on_destroy: could never be
                # triggered by the joblib tests with CPython.
                raise KeyError(obj)
        except KeyError:
            # Insert the new entry in the mapping along with a weakref
            # callback to automatically delete the entry from the mapping
            # as soon as the object used as key is garbage collected.
            def on_destroy(_):
                del self._data[key]
            ref = weakref.ref(obj, on_destroy)
        self._data[key] = ref, value

    def __getstate__(self):
        raise PicklingError("_WeakArrayKeyMap is not pickleable")


###############################################################################
# Support for efficient transient pickling of numpy data structures


def _get_backing_memmap(a):
    """Recursively look up the original np.memmap instance base if any."""
    b = getattr(a, 'base', None)
    if b is None:
        # TODO: check scipy sparse datastructure if scipy is installed
        # a nor its descendants do not have a memmap base
        return None

    elif isinstance(b, mmap):
        # a is already a real memmap instance.
        return a

    else:
        # Recursive exploration of the base ancestry
        return _get_backing_memmap(b)


def _get_temp_dir(pool_folder_name, temp_folder=None):
    """Get the full path to a subfolder inside the temporary folder.

    Parameters
    ----------
    pool_folder_name : str
        Sub-folder name used for the serialization of a pool instance.

    temp_folder: str, optional
        Folder to be used by the pool for memmapping large arrays
        for sharing memory with worker processes. If None, this will try in
        order:

        - a folder pointed by the JOBLIB_TEMP_FOLDER environment
          variable,
        - /dev/shm if the folder exists and is writable: this is a
          RAMdisk filesystem available by default on modern Linux
          distributions,
        - the default system temporary folder that can be
          overridden with TMP, TMPDIR or TEMP environment
          variables, typically /tmp under Unix operating systems.

    Returns
    -------
    pool_folder : str
       full path to the temporary folder
    use_shared_mem : bool
       whether the temporary folder is written to the system shared memory
       folder or some other temporary folder.
    """
    use_shared_mem = False
    if temp_folder is None:
        temp_folder = os.environ.get('JOBLIB_TEMP_FOLDER', None)
    if temp_folder is None:
        if os.path.exists(SYSTEM_SHARED_MEM_FS):
            try:
                shm_stats = os.statvfs(SYSTEM_SHARED_MEM_FS)
                available_nbytes = shm_stats.f_bsize * shm_stats.f_bavail
                if available_nbytes > SYSTEM_SHARED_MEM_FS_MIN_SIZE:
                    # Try to see if we have write access to the shared mem
                    # folder only if it is reasonably large (that is 2GB or
                    # more).
                    temp_folder = SYSTEM_SHARED_MEM_FS
                    pool_folder = os.path.join(temp_folder, pool_folder_name)
                    if not os.path.exists(pool_folder):
                        os.makedirs(pool_folder)
                    use_shared_mem = True
            except (IOError, OSError):
                # Missing rights in the /dev/shm partition, fallback to regular
                # temp folder.
                temp_folder = None
    if temp_folder is None:
        # Fallback to the default tmp folder, typically /tmp
        temp_folder = tempfile.gettempdir()
    temp_folder = os.path.abspath(os.path.expanduser(temp_folder))
    pool_folder = os.path.join(temp_folder, pool_folder_name)
    return pool_folder, use_shared_mem


def has_shareable_memory(a):
    """Return True if a is backed by some mmap buffer directly or not."""
    return _get_backing_memmap(a) is not None


def _strided_from_memmap(filename, dtype, mode, offset, order, shape, strides,
                         total_buffer_len, track):
    """Reconstruct an array view on a memory mapped file."""
    if mode == 'w+':
        # Do not zero the original data when unpickling
        mode = 'r+'

    if strides is None:
        # Simple, contiguous memmap
        return make_memmap(filename, dtype=dtype, shape=shape, mode=mode,
                           offset=offset, order=order, track=track)
    else:
        # For non-contiguous data, memmap the total enclosing buffer and then
        # extract the non-contiguous view with the stride-tricks API
        base = make_memmap(filename, dtype=dtype, shape=total_buffer_len,
                           mode=mode, offset=offset, order=order, track=track)
        return as_strided(base, shape=shape, strides=strides)


def _reduce_memmap_backed(a, m):
    """Pickling reduction for memmap backed arrays.

    a is expected to be an instance of np.ndarray (or np.memmap)
    m is expected to be an instance of np.memmap on the top of the ``base``
    attribute ancestry of a. ``m.base`` should be the real python mmap object.
    """
    # offset that comes from the striding differences between a and m
    util.debug('[MEMMAP REDUCE] reducing a memmap-backed array '
               '(shape, {}, pid: {})'.format(a.shape, os.getpid()))
    a_start, a_end = np.byte_bounds(a)
    m_start = np.byte_bounds(m)[0]
    offset = a_start - m_start

    # offset from the backing memmap
    offset += m.offset

    if m.flags['F_CONTIGUOUS']:
        order = 'F'
    else:
        # The backing memmap buffer is necessarily contiguous hence C if not
        # Fortran
        order = 'C'

    if a.flags['F_CONTIGUOUS'] or a.flags['C_CONTIGUOUS']:
        # If the array is a contiguous view, no need to pass the strides
        strides = None
        total_buffer_len = None
    else:
        # Compute the total number of items to map from which the strided
        # view will be extracted.
        strides = a.strides
        total_buffer_len = (a_end - a_start) // a.itemsize

    track = False
    if m.filename in JOBLIB_MMAPS:
        # Only the mmap objects created by joblib should be tracked by
        # the resource_tracker.
        resource_tracker.register(m.filename, "file")
        track = True

    return (_strided_from_memmap,
            (m.filename, a.dtype, m.mode, offset, order, a.shape, strides,
             total_buffer_len, track))


def reduce_memmap(a):
    """Pickle the descriptors of a memmap instance to reopen on same file."""
    m = _get_backing_memmap(a)
    if m is not None:
        # m is a real mmap backed memmap instance, reduce a preserving striding
        # information
        return _reduce_memmap_backed(a, m)
    else:
        # This memmap instance is actually backed by a regular in-memory
        # buffer: this can happen when using binary operators on numpy.memmap
        # instances
        return (loads, (dumps(np.asarray(a), protocol=HIGHEST_PROTOCOL),))


class ArrayMemmapReducer(object):
    """Reducer callable to dump large arrays to memmap files.

    Parameters
    ----------
    max_nbytes: int
        Threshold to trigger memmapping of large arrays to files created
        a folder.
    temp_folder: str
        Path of a folder where files for backing memmapped arrays are created.
    mmap_mode: 'r', 'r+' or 'c'
        Mode for the created memmap datastructure. See the documentation of
        numpy.memmap for more details. Note: 'w+' is coerced to 'r+'
        automatically to avoid zeroing the data on unpickling.
    verbose: int, optional, 0 by default
        If verbose > 0, memmap creations are logged.
        If verbose > 1, both memmap creations, reuse and array pickling are
        logged.
    prewarm: bool, optional, False by default.
        Force a read on newly memmapped array to make sure that OS pre-cache it
        memory. This can be useful to avoid concurrent disk access when the
        same data array is passed to different worker processes.
    """

    def __init__(self, max_nbytes, temp_folder, mmap_mode, verbose=0,
                 prewarm=True):
        self._max_nbytes = max_nbytes
        self._temp_folder = temp_folder
        self._mmap_mode = mmap_mode
        self.verbose = int(verbose)
        self._prewarm = prewarm
        self._memmaped_arrays = _WeakArrayKeyMap()

    def __reduce__(self):
        # The ArrayMemmapReducer is passed to the children processes: it needs
        # to be pickled but the _WeakArrayKeyMap need to be skipped as it's
        # only guaranteed to be consistent with the parent process memory
        # garbage collection.
        args = (self._max_nbytes, self._temp_folder, self._mmap_mode)
        kwargs = {
            'verbose': self.verbose,
            'prewarm': self._prewarm,
        }
        return ArrayMemmapReducer, args, kwargs

    def __call__(self, a):
        m = _get_backing_memmap(a)
        if m is not None and isinstance(m, np.memmap):
            # a is already backed by a memmap file, let's reuse it directly
            util.debug(
                '[MEMMAP REDUCE] reducing mmemmap (shape, {}, pid: {})'.format(
                    a.shape, os.getpid())
            )
            return _reduce_memmap_backed(a, m)

        if (not a.dtype.hasobject and self._max_nbytes is not None and
                a.nbytes > self._max_nbytes):
            # check that the folder exists (lazily create the pool temp folder
            # if required)
            try:
                os.makedirs(self._temp_folder)
                os.chmod(self._temp_folder, FOLDER_PERMISSIONS)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e

            try:
                basename = self._memmaped_arrays.get(a)
            except KeyError:
                # Generate a new unique random filename. The process and thread
                # ids are only useful for debugging purpose and to make it
                # easier to cleanup orphaned files in case of hard process
                # kill (e.g. by "kill -9" or segfault).
                basename = "{}-{}-{}.pkl".format(
                    os.getpid(), id(threading.current_thread()), uuid4().hex)
                self._memmaped_arrays.set(a, basename)
            filename = os.path.join(self._temp_folder, basename)

            # In case the same array with the same content is passed several
            # times to the pool subprocess children, serialize it only once

            is_new_memmap = filename not in JOBLIB_MMAPS

            # add the memmap to the list of temporary memmaps created by joblib
            JOBLIB_MMAPS.add(filename)
            resource_tracker.register(filename, "file")

            if is_new_memmap:
                # Incref each temporary memmap created by joblib one extra
                # time.  This means that these memmaps will only be deleted
                # once an extra maybe_unlink() is called, which is done when a
                # Parallel object returns
                resource_tracker.register(filename, "file")

            if not os.path.exists(filename):
                util.debug(
                    "[ARRAY DUMP] Pickling new array (shape={}, dtype={}) "
                    "creating a new memmap".format(
                        a.shape, a.dtype, os.path.basename(filename)))
                for dumped_filename in dump(a, filename):
                    os.chmod(dumped_filename, FILE_PERMISSIONS)

                if self._prewarm:
                    # Warm up the data by accessing it. This operation ensures
                    # that the disk access required to create the memmapping
                    # file are performed in the reducing process and avoids
                    # concurrent memmap creation in multiple children
                    # processes.
                    load(filename, mmap_mode=self._mmap_mode).max()

            else:
                util.debug(
                    "[ARRAY DUMP] Pickling known array (shape={}, dtype={}) "
                    "retrieving memmap {}".format(
                        a.shape, a.dtype, os.path.basename(filename)))

            # The worker process will use joblib.load to memmap the data
            return (
                (load_temporary_memmap, (filename, self._mmap_mode, True))
            )
        else:
            # do not convert a into memmap, let pickler do its usual copy with
            # the default system pickler
            util.debug(
                '[ARRAY DUMP] Pickling array (NO MEMMAPPING) (shape={}, '
                ' dtype={}).'.format(a.shape, a.dtype))
            return (loads, (dumps(a, protocol=HIGHEST_PROTOCOL),))


def get_memmapping_reducers(
        pool_id, forward_reducers=None, backward_reducers=None,
        temp_folder=None, max_nbytes=1e6, mmap_mode='r', verbose=0,
        prewarm=False, **kwargs):
    """Construct a pair of memmapping reducer linked to a tmpdir.

    This function manage the creation and the clean up of the temporary folders
    underlying the memory maps and should be use to get the reducers necessary
    to construct joblib pool or executor.
    """
    if forward_reducers is None:
        forward_reducers = dict()
    if backward_reducers is None:
        backward_reducers = dict()

    # Prepare a sub-folder name for the serialization of this particular
    # pool instance (do not create in advance to spare FS write access if
    # no array is to be dumped):
    pool_folder_name = "joblib_memmapping_folder_{}_{}".format(
        os.getpid(), pool_id)
    pool_folder, use_shared_mem = _get_temp_dir(pool_folder_name,
                                                temp_folder)

    # Register the garbage collector at program exit in case caller forgets
    # to call terminate explicitly: note we do not pass any reference to
    # self to ensure that this callback won't prevent garbage collection of
    # the pool instance and related file handler resources such as POSIX
    # semaphores and pipes
    pool_module_name = whichmodule(delete_folder, 'delete_folder')
    resource_tracker.register(pool_folder, "folder")

    def _cleanup():
        # In some cases the Python runtime seems to set delete_folder to
        # None just before exiting when accessing the delete_folder
        # function from the closure namespace. So instead we reimport
        # the delete_folder function explicitly.
        # https://github.com/joblib/joblib/issues/328
        # We cannot just use from 'joblib.pool import delete_folder'
        # because joblib should only use relative imports to allow
        # easy vendoring.
        delete_folder = __import__(
            pool_module_name, fromlist=['delete_folder']).delete_folder
        try:
            delete_folder(pool_folder)
            resource_tracker.unregister(pool_folder, "folder")
        except WindowsError:
            warnings.warn("Failed to clean temporary folder: {}"
                          .format(pool_folder))

    atexit.register(_cleanup)

    if np is not None:
        # Register smart numpy.ndarray reducers that detects memmap backed
        # arrays and that is also able to dump to memmap large in-memory
        # arrays over the max_nbytes threshold
        if prewarm == "auto":
            prewarm = not use_shared_mem
        forward_reduce_ndarray = ArrayMemmapReducer(
            max_nbytes, pool_folder, mmap_mode, verbose,
            prewarm=prewarm)
        forward_reducers[np.ndarray] = forward_reduce_ndarray
        forward_reducers[np.memmap] = reduce_memmap

        # Communication from child process to the parent process always
        # pickles in-memory numpy.ndarray without dumping them as memmap
        # to avoid confusing the caller and make it tricky to collect the
        # temporary folder
        backward_reduce_ndarray = ArrayMemmapReducer(
            None, pool_folder, mmap_mode, verbose)
        backward_reducers[np.ndarray] = backward_reduce_ndarray
        backward_reducers[np.memmap] = reduce_memmap

    return forward_reducers, backward_reducers, pool_folder
