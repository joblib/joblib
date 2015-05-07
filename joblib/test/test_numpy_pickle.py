"""
Test the numpy pickler as a replacement of the standard pickler.

"""

from tempfile import mkdtemp
import copy
import shutil
import os
import random
import sys
import re
import tempfile

import nose

from joblib.test.common import np, with_numpy

# numpy_pickle is not a drop-in replacement of pickle, as it takes
# filenames instead of open files as arguments.
from joblib import numpy_pickle
from joblib.test import data

###############################################################################
# Define a list of standard types.
# Borrowed from dill, initial author: Micheal McKerns:
# http://dev.danse.us/trac/pathos/browser/dill/dill_test2.py

typelist = []

# testing types
_none = None
typelist.append(_none)
_type = type
typelist.append(_type)
_bool = bool(1)
typelist.append(_bool)
_int = int(1)
typelist.append(_int)
try:
    _long = long(1)
    typelist.append(_long)
except NameError:
    # long is not defined in python 3
    pass
_float = float(1)
typelist.append(_float)
_complex = complex(1)
typelist.append(_complex)
_string = str(1)
typelist.append(_string)
try:
    _unicode = unicode(1)
    typelist.append(_unicode)
except NameError:
    # unicode is not defined in python 3
    pass
_tuple = ()
typelist.append(_tuple)
_list = []
typelist.append(_list)
_dict = {}
typelist.append(_dict)
try:
    _file = file
    typelist.append(_file)
except NameError:
    pass # file does not exists in Python 3
try:
    _buffer = buffer
    typelist.append(_buffer)
except NameError:
    # buffer does not exists in Python 3
    pass
_builtin = len
typelist.append(_builtin)


def _function(x):
    yield x


class _class:
    def _method(self):
        pass


class _newclass(object):
    def _method(self):
        pass


typelist.append(_function)
typelist.append(_class)
typelist.append(_newclass)  # <type 'type'>
_instance = _class()
typelist.append(_instance)
_object = _newclass()
typelist.append(_object)  # <type 'class'>


###############################################################################
# Test fixtures

env = dict()


def setup_module():
    """ Test setup.
    """
    env['dir'] = mkdtemp()
    env['filename'] = os.path.join(env['dir'], 'test.pkl')
    print(80 * '_')
    print('setup numpy_pickle')
    print(80 * '_')


def teardown_module():
    """ Test teardown.
    """
    shutil.rmtree(env['dir'])
    #del env['dir']
    #del env['filename']
    print(80 * '_')
    print('teardown numpy_pickle')
    print(80 * '_')


###############################################################################
# Tests

def test_standard_types():
    # Test pickling and saving with standard types.
    filename = env['filename']
    for compress in [0, 1]:
        for member in typelist:
            # Change the file name to avoid side effects between tests
            this_filename = filename + str(random.randint(0, 1000))
            numpy_pickle.dump(member, this_filename, compress=compress)
            _member = numpy_pickle.load(this_filename)
            # We compare the pickled instance to the reloaded one only if it
            # can be compared to a copied one
            if member == copy.deepcopy(member):
                yield nose.tools.assert_equal, member, _member


def test_value_error():
    # Test inverting the input arguments to dump
    nose.tools.assert_raises(ValueError, numpy_pickle.dump, 'foo',
                             dict())


@with_numpy
def test_numpy_persistence():
    filename = env['filename']
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))
    for compress, cache_size in ((0, 0), (1, 0), (1, 10)):
        # We use 'a.T' to have a non C-contiguous array.
        for index, obj in enumerate(((a,), (a.T,), (a, a), [a, a, a])):
            # Change the file name to avoid side effects between tests
            this_filename = filename + str(random.randint(0, 1000))
            filenames = numpy_pickle.dump(obj, this_filename,
                                          compress=compress,
                                          cache_size=cache_size)
            # Check that one file was created per array
            if not compress:
                nose.tools.assert_equal(len(filenames), len(obj) + 1)
            # Check that these files do exist
            for file in filenames:
                nose.tools.assert_true(
                    os.path.exists(os.path.join(env['dir'], file)))

            # Unpickle the object
            obj_ = numpy_pickle.load(this_filename)
            # Check that the items are indeed arrays
            for item in obj_:
                nose.tools.assert_true(isinstance(item, np.ndarray))
            # And finally, check that all the values are equal.
            nose.tools.assert_true(np.all(np.array(obj) ==
                                                np.array(obj_)))

        # Now test with array subclasses
        for obj in (
                    np.matrix(np.zeros(10)),
                    np.core.multiarray._reconstruct(np.memmap, (), np.float)
                   ):
            this_filename = filename + str(random.randint(0, 1000))
            filenames = numpy_pickle.dump(obj, this_filename,
                                          compress=compress,
                                          cache_size=cache_size)
            obj_ = numpy_pickle.load(this_filename)
            if (type(obj) is not np.memmap
                        and hasattr(obj, '__array_prepare__')):
                # We don't reconstruct memmaps
                nose.tools.assert_true(isinstance(obj_, type(obj)))

    # Finally smoke test the warning in case of compress + mmap_mode
    this_filename = filename + str(random.randint(0, 1000))
    numpy_pickle.dump(a, this_filename, compress=1)
    numpy_pickle.load(this_filename, mmap_mode='r')


@with_numpy
def test_memmap_persistence():
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')

    nose.tools.assert_true(isinstance(b, np.memmap))


@with_numpy
def test_memmap_persistence_mixed_dtypes():
    # loading datastructures that have sub-arrays with dtype=object
    # should not prevent memmaping on fixed size dtype sub-arrays.
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    b = np.array([1, 'b'], dtype=object)
    construct = (a, b)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(construct, filename)
    a_clone, b_clone = numpy_pickle.load(filename, mmap_mode='r')

    # the floating point array has been memory mapped
    nose.tools.assert_true(isinstance(a_clone, np.memmap))

    # the object-dtype array has been loaded in memory
    nose.tools.assert_false(isinstance(b_clone, np.memmap))


@with_numpy
def test_masked_array_persistence():
    # The special-case picker fails, because saving masked_array
    # not implemented, but it just delegates to the standard pickler.
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    a = np.ma.masked_greater(a, 0.5)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')
    nose.tools.assert_true(isinstance(b, np.ma.masked_array))


def test_z_file():
    # Test saving and loading data with Zfiles
    filename = env['filename'] + str(random.randint(0, 1000))
    data = numpy_pickle.asbytes('Foo, \n Bar, baz, \n\nfoobar')
    with open(filename, 'wb') as f:
        numpy_pickle.write_zfile(f, data)
    with open(filename, 'rb') as f:
        data_read = numpy_pickle.read_zfile(f)
    nose.tools.assert_equal(data, data_read)


@with_numpy
def test_compressed_pickle_dump_and_load():
    expected_list = [np.arange(5, dtype=np.int64),
                     np.arange(5, dtype=np.float64),
                     # .tostring actually returns bytes and is a
                     # compatibility alias for .tobytes which was
                     # added in 1.9.0
                     np.arange(256, dtype=np.uint8).tostring(),
                     u"C'est l'\xe9t\xe9 !"]

    with tempfile.NamedTemporaryFile(suffix='.gz', dir=env['dir']) as f:
        fname = f.name

    try:
        numpy_pickle.dump(expected_list, fname, compress=1)
        result_list = numpy_pickle.load(fname)
        for result, expected in zip(result_list, expected_list):
            if isinstance(expected, np.ndarray):
                nose.tools.assert_equal(result.dtype, expected.dtype)
                np.testing.assert_equal(result, expected)
            else:
                nose.tools.assert_equal(result, expected)
    finally:
        os.remove(fname)


@with_numpy
def test_compressed_pickle_python_2_3_compatibility():
    expected_list = [np.arange(5, dtype=np.int64),
                     np.arange(5, dtype=np.float64),
                     # .tostring actually returns bytes and is a
                     # compatibility alias for .tobytes which was
                     # added in 1.9.0
                     np.arange(256, dtype=np.uint8).tostring(),
                     u"C'est l'\xe9t\xe9 !"]

    test_data_dir = os.path.dirname(os.path.abspath(data.__file__))
    # These files have been generated with the
    # joblib/test/data/create_numpy_pickle.py script for the relevant
    # python and joblib versions
    basenames = ['joblib_0.8.4_compressed_pickle_py27.gz',
                 'joblib_0.9.0_compressed_pickle_py27.gz',
                 'joblib_0.8.4_compressed_pickle_py33.gz',
                 'joblib_0.9.0_compressed_pickle_py33.gz',
                 'joblib_0.8.4_compressed_pickle_py34.gz',
                 'joblib_0.9.0_compressed_pickle_py34.gz']
    data_filenames = [os.path.join(test_data_dir, bname)
                      for bname in basenames]

    for fname in data_filenames:
        version_match = re.match(r'.+py(\d)(\d).gz', fname)
        python_version_used_for_writing = tuple(
            [int(each) for each in version_match.groups()])
        python_version_used_for_reading = sys.version_info[:2]

        python_version_to_default_pickle_protocol = {
            (2, 6): 2, (2, 7): 2,
            (3, 0): 3, (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 4}

        pickle_reading_protocol = python_version_to_default_pickle_protocol[
            python_version_used_for_reading]
        pickle_writing_protocol = python_version_to_default_pickle_protocol[
            python_version_used_for_writing]
        if ('0.8.4' not in fname or
                pickle_reading_protocol >=
                pickle_writing_protocol):
            result_list = numpy_pickle.load(fname)
            for result, expected in zip(result_list, expected_list):
                if isinstance(expected, np.ndarray):
                    nose.tools.assert_equal(result.dtype, expected.dtype)
                    np.testing.assert_equal(result, expected)
                else:
                    nose.tools.assert_equal(result, expected)
        else:
            # For joblib <= 0.8.4 compressed pickles written with
            # python `version = v` can not be read by python with
            # `version < v' because of differences in the default
            # pickle protocol (2 for python 2, 3 for python 3.3 and 4
            # for python 3.4)
            try:
                numpy_pickle.load(fname)
                raise AssertionError('Numpy pickle loading should '
                                     'have raised a ValueError exception')
            except ValueError as e:
                nose.tools.assert_true(
                    'unsupported pickle protocol' in str(e.args))


################################################################################
# Test dumping array subclasses
if np is not None:

    class SubArray(np.ndarray):

        def __reduce__(self):
            return (_load_sub_array, (np.asarray(self), ))


    def _load_sub_array(arr):
        d = SubArray(arr.shape)
        d[:] = arr
        return d

@with_numpy
def test_numpy_subclass():
    filename = env['filename']
    a = SubArray((10,))
    numpy_pickle.dump(a, filename)
    c = numpy_pickle.load(filename)
    nose.tools.assert_true(isinstance(c, SubArray))
