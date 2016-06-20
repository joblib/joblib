"""Test the numpy pickler as a replacement of the standard pickler."""

from tempfile import mkdtemp
import copy
import shutil
import os
import random
import sys
import re
import tempfile
import io
import warnings
import nose
import gzip
import zlib
import bz2
import pickle
from contextlib import closing

from joblib.test.common import np, with_numpy
from joblib.test.common import with_memory_profiler, memory_used
from joblib.testing import assert_raises_regex

# numpy_pickle is not a drop-in replacement of pickle, as it takes
# filenames instead of open files as arguments.
from joblib import numpy_pickle
from joblib.test import data

from joblib._compat import PY3_OR_LATER, PY26
from joblib.numpy_pickle_utils import _IO_BUFFER_SIZE, BinaryZlibFile
from joblib.numpy_pickle_utils import _detect_compressor, _COMPRESSORS

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
    pass  # file does not exists in Python 3
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
    # del env['dir']
    # del env['filename']
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


def test_compress_level_error():
    # Verify that passing an invalid compress argument raises an error.
    wrong_compress = (-1, 10, 'wrong')
    for wrong in wrong_compress:
        exception_msg = 'Non valid compress level given: "{0}"'.format(wrong)
        assert_raises_regex(ValueError,
                            exception_msg,
                            numpy_pickle.dump, 'dummy', 'foo', compress=wrong)


@with_numpy
def test_numpy_persistence():
    filename = env['filename']
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))
    for compress in (False, True, 0, 3):
        # We use 'a.T' to have a non C-contiguous array.
        for index, obj in enumerate(((a,), (a.T,), (a, a), [a, a, a])):
            # Change the file name to avoid side effects between tests
            this_filename = filename + str(random.randint(0, 1000))

            filenames = numpy_pickle.dump(obj, this_filename,
                                          compress=compress)

            # All is cached in one file
            nose.tools.assert_equal(len(filenames), 1)
            # Check that only one file was created
            nose.tools.assert_equal(filenames[0], this_filename)
            # Check that this file does exist
            nose.tools.assert_true(
                os.path.exists(os.path.join(env['dir'], filenames[0])))

            # Unpickle the object
            obj_ = numpy_pickle.load(this_filename)
            # Check that the items are indeed arrays
            for item in obj_:
                nose.tools.assert_true(isinstance(item, np.ndarray))
            # And finally, check that all the values are equal.
            np.testing.assert_array_equal(np.array(obj), np.array(obj_))

        # Now test with array subclasses
        for obj in (np.matrix(np.zeros(10)),
                    np.memmap(filename + str(random.randint(0, 1000)) + 'mmap',
                              mode='w+', shape=4, dtype=np.float)):
            this_filename = filename + str(random.randint(0, 1000))
            filenames = numpy_pickle.dump(obj, this_filename,
                                          compress=compress)
            # All is cached in one file
            nose.tools.assert_equal(len(filenames), 1)

            obj_ = numpy_pickle.load(this_filename)
            if (type(obj) is not np.memmap and
                    hasattr(obj, '__array_prepare__')):
                # We don't reconstruct memmaps
                nose.tools.assert_true(isinstance(obj_, type(obj)))

            np.testing.assert_array_equal(obj_, obj)

        # Test with an object containing multiple numpy arrays
        obj = ComplexTestObject()
        filenames = numpy_pickle.dump(obj, this_filename,
                                      compress=compress)
        # All is cached in one file
        nose.tools.assert_equal(len(filenames), 1)

        obj_loaded = numpy_pickle.load(this_filename)
        nose.tools.assert_true(isinstance(obj_loaded, type(obj)))
        np.testing.assert_array_equal(obj_loaded.array_float, obj.array_float)
        np.testing.assert_array_equal(obj_loaded.array_int, obj.array_int)
        np.testing.assert_array_equal(obj_loaded.array_obj, obj.array_obj)


@with_numpy
def test_numpy_persistence_bufferred_array_compression():
    big_array = np.ones((_IO_BUFFER_SIZE + 100), dtype=np.uint8)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(big_array, filename, compress=True)
    arr_reloaded = numpy_pickle.load(filename)

    np.testing.assert_array_equal(big_array, arr_reloaded)


@with_numpy
def test_memmap_persistence():
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')

    nose.tools.assert_true(isinstance(b, np.memmap))

    # Test with an object containing multiple numpy arrays
    filename = env['filename'] + str(random.randint(0, 1000))
    obj = ComplexTestObject()
    numpy_pickle.dump(obj, filename)
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r')
    nose.tools.assert_true(isinstance(obj_loaded, type(obj)))
    nose.tools.assert_true(isinstance(obj_loaded.array_float, np.memmap))
    nose.tools.assert_false(obj_loaded.array_float.flags.writeable)
    nose.tools.assert_true(isinstance(obj_loaded.array_int, np.memmap))
    nose.tools.assert_false(obj_loaded.array_int.flags.writeable)
    # Memory map not allowed for numpy object arrays
    nose.tools.assert_false(isinstance(obj_loaded.array_obj, np.memmap))
    np.testing.assert_array_equal(obj_loaded.array_float,
                                  obj.array_float)
    np.testing.assert_array_equal(obj_loaded.array_int,
                                  obj.array_int)
    np.testing.assert_array_equal(obj_loaded.array_obj,
                                  obj.array_obj)

    # Test we can write in memmaped arrays
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r+')
    nose.tools.assert_true(obj_loaded.array_float.flags.writeable)
    obj_loaded.array_float[0:10] = 10.0
    nose.tools.assert_true(obj_loaded.array_int.flags.writeable)
    obj_loaded.array_int[0:10] = 10

    obj_reloaded = numpy_pickle.load(filename, mmap_mode='r')
    np.testing.assert_array_equal(obj_reloaded.array_float,
                                  obj_loaded.array_float)
    np.testing.assert_array_equal(obj_reloaded.array_int,
                                  obj_loaded.array_int)

    # Test w+ mode is caught and the mode has switched to r+
    numpy_pickle.load(filename, mmap_mode='w+')
    nose.tools.assert_true(obj_loaded.array_int.flags.writeable)
    nose.tools.assert_equal(obj_loaded.array_int.mode, 'r+')
    nose.tools.assert_true(obj_loaded.array_float.flags.writeable)
    nose.tools.assert_equal(obj_loaded.array_float.mode, 'r+')


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


@with_numpy
def test_compress_mmap_mode_warning():
    # Test the warning in case of compress + mmap_mode
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    this_filename = env['filename'] + str(random.randint(0, 1000))
    numpy_pickle.dump(a, this_filename, compress=1)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        numpy_pickle.load(this_filename, mmap_mode='r+')
        nose.tools.assert_equal(len(caught_warnings), 1)
        for warn in caught_warnings:
            nose.tools.assert_equal(warn.category, UserWarning)
            nose.tools.assert_equal(warn.message.args[0],
                                    'File "%(filename)s" is compressed using '
                                    '"%(compressor)s" which is not compatible '
                                    'with mmap_mode "%(mmap_mode)s" flag '
                                    'passed. mmap_mode option will be '
                                    'ignored.' % {'filename': this_filename,
                                                  'mmap_mode': 'r+',
                                                  'compressor': 'zlib'})


@with_numpy
def test_cache_size_warning():
    # Check deprecation warning raised when cache size is not None
    filename = env['filename'] + str(random.randint(0, 1000))
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))

    for cache_size in (None, 0, 10):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            numpy_pickle.dump(a, filename, cache_size=cache_size)
            expected_nb_warnings = 1 if cache_size is not None else 0
            nose.tools.assert_equal(len(caught_warnings),
                                    expected_nb_warnings)
            for warn in caught_warnings:
                nose.tools.assert_equal(warn.category, DeprecationWarning)
                nose.tools.assert_equal(warn.message.args[0],
                                        "Please do not set 'cache_size' in "
                                        "joblib.dump, this parameter has no "
                                        "effect and will be removed. "
                                        "You used 'cache_size={0}'".format(
                                            cache_size))


@with_numpy
@with_memory_profiler
def test_memory_usage():
    # Verify memory stays within expected bounds.
    filename = env['filename']
    small_array = np.ones((10, 10))
    big_array = np.ones(shape=100 * int(1e6), dtype=np.uint8)
    small_matrix = np.matrix(small_array)
    big_matrix = np.matrix(big_array)
    for compress in (True, False):
        for obj in (small_array, big_array, small_matrix, big_matrix):
            size = obj.nbytes / 1e6
            obj_filename = filename + str(np.random.randint(0, 1000))
            mem_used = memory_used(numpy_pickle.dump,
                                   obj, obj_filename, compress=compress)

            # The memory used to dump the object shouldn't exceed the buffer
            # size used to write array chunks (16MB).
            write_buf_size = _IO_BUFFER_SIZE + 16 * 1024 ** 2 / 1e6
            nose.tools.assert_true(mem_used <= write_buf_size)

            mem_used = memory_used(numpy_pickle.load, obj_filename)
            # memory used should be less than array size + buffer size used to
            # read the array chunk by chunk.
            read_buf_size = 32 + _IO_BUFFER_SIZE  # MiB
            nose.tools.assert_true(mem_used < size + read_buf_size)


@with_numpy
def test_compressed_pickle_dump_and_load():
    expected_list = [np.arange(5, dtype=np.dtype('<i8')),
                     np.arange(5, dtype=np.dtype('>i8')),
                     np.arange(5, dtype=np.dtype('<f8')),
                     np.arange(5, dtype=np.dtype('>f8')),
                     np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'),
                     # .tostring actually returns bytes and is a
                     # compatibility alias for .tobytes which was
                     # added in 1.9.0
                     np.arange(256, dtype=np.uint8).tostring(),
                     # np.matrix is a subclass of np.ndarray, here we want
                     # to verify this type of object is correctly unpickled
                     # among versions.
                     np.matrix([0, 1, 2], dtype=np.dtype('<i8')),
                     np.matrix([0, 1, 2], dtype=np.dtype('>i8')),
                     u"C'est l'\xe9t\xe9 !"]

    with tempfile.NamedTemporaryFile(suffix='.gz', dir=env['dir']) as f:
        fname = f.name

    try:
        dumped_filenames = numpy_pickle.dump(expected_list, fname, compress=1)
        nose.tools.assert_equal(len(dumped_filenames), 1)
        result_list = numpy_pickle.load(fname)
        for result, expected in zip(result_list, expected_list):
            if isinstance(expected, np.ndarray):
                nose.tools.assert_equal(result.dtype, expected.dtype)
                np.testing.assert_equal(result, expected)
            else:
                nose.tools.assert_equal(result, expected)
    finally:
        os.remove(fname)


def _check_pickle(filename, expected_list):
    """Helper function to test joblib pickle content.

    Note: currently only pickles containing an iterable are supported
    by this function.
    """
    if (not PY3_OR_LATER and (filename.endswith('.xz') or
                              filename.endswith('.lzma'))):
        # lzma is not supported for python versions < 3.3
        nose.tools.assert_raises(NotImplementedError,
                                 numpy_pickle.load, filename)
        return

    version_match = re.match(r'.+py(\d)(\d).+', filename)
    py_version_used_for_writing = int(version_match.group(1))
    py_version_used_for_reading = sys.version_info[0]

    py_version_to_default_pickle_protocol = {2: 2, 3: 3}
    pickle_reading_protocol = py_version_to_default_pickle_protocol.get(
        py_version_used_for_reading, 4)
    pickle_writing_protocol = py_version_to_default_pickle_protocol.get(
        py_version_used_for_writing, 4)
    if pickle_reading_protocol >= pickle_writing_protocol:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                result_list = numpy_pickle.load(filename)
                expected_nb_warnings = 1 if ("0.9" in filename or
                                             "0.8.4" in filename) else 0
                nose.tools.assert_equal(len(caught_warnings),
                                        expected_nb_warnings)
            for warn in caught_warnings:
                nose.tools.assert_equal(warn.category, DeprecationWarning)
                nose.tools.assert_equal(warn.message.args[0],
                                        "The file '{0}' has been generated "
                                        "with a joblib version less than "
                                        "0.10. Please regenerate this pickle "
                                        "file.".format(filename))
            for result, expected in zip(result_list, expected_list):
                if isinstance(expected, np.ndarray):
                    nose.tools.assert_equal(result.dtype, expected.dtype)
                    np.testing.assert_equal(result, expected)
                else:
                    nose.tools.assert_equal(result, expected)
        except Exception as exc:
            # When trying to read with python 3 a pickle generated
            # with python 2 we expect a user-friendly error
            if (py_version_used_for_reading == 3 and
                    py_version_used_for_writing == 2):
                nose.tools.assert_true(isinstance(exc, ValueError))
                message = ('You may be trying to read with '
                           'python 3 a joblib pickle generated with python 2.')
                nose.tools.assert_true(message in str(exc))
            else:
                raise
    else:
        # Pickle protocol used for writing is too high. We expect a
        # "unsupported pickle protocol" error message
        try:
            numpy_pickle.load(filename)
            raise AssertionError('Numpy pickle loading should '
                                 'have raised a ValueError exception')
        except ValueError as e:
            message = 'unsupported pickle protocol: {0}'.format(
                pickle_writing_protocol)
            nose.tools.assert_true(message in str(e.args))


@with_numpy
def test_joblib_pickle_across_python_versions():
    # We need to be specific about dtypes in particular endianness
    # because the pickles can be generated on one architecture and
    # the tests run on another one. See
    # https://github.com/joblib/joblib/issues/279.
    expected_list = [np.arange(5, dtype=np.dtype('<i8')),
                     np.arange(5, dtype=np.dtype('<f8')),
                     np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'),
                     # .tostring actually returns bytes and is a
                     # compatibility alias for .tobytes which was
                     # added in 1.9.0
                     np.arange(256, dtype=np.uint8).tostring(),
                     # np.matrix is a subclass of np.ndarray, here we want
                     # to verify this type of object is correctly unpickled
                     # among versions.
                     np.matrix([0, 1, 2], dtype=np.dtype('<i8')),
                     u"C'est l'\xe9t\xe9 !"]

    # Testing all the compressed and non compressed
    # pickles in joblib/test/data. These pickles were generated by
    # the joblib/test/data/create_numpy_pickle.py script for the
    # relevant python, joblib and numpy versions.
    test_data_dir = os.path.dirname(os.path.abspath(data.__file__))

    pickle_extensions = ('.pkl', '.gz', '.gzip', '.bz2', '.xz', '.lzma')
    pickle_filenames = [os.path.join(test_data_dir, fn)
                        for fn in os.listdir(test_data_dir)
                        if any(fn.endswith(ext) for ext in pickle_extensions)]

    for fname in pickle_filenames:
        _check_pickle(fname, expected_list)


def test_compress_tuple_argument():
    compress_tuples = (('zlib', 3),
                       ('gzip', 3))

    # Verify the tuple is correctly taken into account.
    filename = env['filename'] + str(random.randint(0, 1000))
    for compress in compress_tuples:
        numpy_pickle.dump("dummy", filename,
                          compress=compress)
        # Verify the file contains the right magic number
        with open(filename, 'rb') as f:
            nose.tools.assert_equal(_detect_compressor(f), compress[0])

    # Verify setting a wrong compress tuple raises a ValueError.
    assert_raises_regex(ValueError,
                        'Compress argument tuple should contain exactly '
                        '2 elements',
                        numpy_pickle.dump, "dummy", filename,
                        compress=('zlib', 3, 'extra'))

    # Verify a tuple with a wrong compress method raises a ValueError.
    msg = 'Non valid compression method given: "{0}"'.format('wrong')
    assert_raises_regex(ValueError, msg,
                        numpy_pickle.dump, "dummy", filename,
                        compress=('wrong', 3))

    # Verify a tuple with a wrong compress level raises a ValueError.
    msg = 'Non valid compress level given: "{0}"'.format('wrong')
    assert_raises_regex(ValueError, msg,
                        numpy_pickle.dump, "dummy", filename,
                        compress=('zlib', 'wrong'))


@with_numpy
def test_joblib_compression_formats():
    compresslevels = (1, 3, 6)
    filename = env['filename'] + str(random.randint(0, 1000))
    objects = (np.ones(shape=(100, 100), dtype='f8'),
               range(10),
               {'a': 1, 2: 'b'}, [], (), {}, 0, 1.0)

    for compress in compresslevels:
        for cmethod in _COMPRESSORS:
            dump_filename = filename + "." + cmethod
            for obj in objects:
                if not PY3_OR_LATER and cmethod in ('xz', 'lzma'):
                    # Lzma module only available for python >= 3.3
                    msg = "{0} compression is only available".format(cmethod)
                    assert_raises_regex(NotImplementedError, msg,
                                        numpy_pickle.dump, obj, dump_filename,
                                        compress=(cmethod, compress))
                else:
                    numpy_pickle.dump(obj, dump_filename,
                                      compress=(cmethod, compress))
                    # Verify the file contains the right magic number
                    with open(dump_filename, 'rb') as f:
                        nose.tools.assert_equal(
                            _detect_compressor(f), cmethod)
                    # Verify the reloaded object is correct
                    obj_reloaded = numpy_pickle.load(dump_filename)
                    nose.tools.assert_true(isinstance(obj_reloaded, type(obj)))
                    if isinstance(obj, np.ndarray):
                        np.testing.assert_array_equal(obj_reloaded, obj)
                    else:
                        nose.tools.assert_equal(obj_reloaded, obj)
                    os.remove(dump_filename)


def _gzip_file_decompress(source_filename, target_filename):
    """Decompress a gzip file."""
    with closing(gzip.GzipFile(source_filename, "rb")) as fo:
        buf = fo.read()

    with open(target_filename, "wb") as fo:
        fo.write(buf)


def _zlib_file_decompress(source_filename, target_filename):
    """Decompress a zlib file."""
    with open(source_filename, 'rb') as fo:
        buf = zlib.decompress(fo.read())

    with open(target_filename, 'wb') as fo:
        fo.write(buf)


def test_load_externally_decompressed_files():
    # Test that BinaryZlibFile generates valid gzip and zlib compressed files.
    obj = "a string to persist"
    filename_raw = env['filename'] + str(random.randint(0, 1000))
    compress_list = (('.z', _zlib_file_decompress),
                     ('.gz', _gzip_file_decompress))

    for extension, decompress in compress_list:
        filename_compressed = filename_raw + extension
        # Use automatic extension detection to compress with the right method.
        numpy_pickle.dump(obj, filename_compressed)

        # Decompress with the corresponding method
        decompress(filename_compressed, filename_raw)

        # Test that the uncompressed pickle can be loaded and
        # that the result is correct.
        obj_reloaded = numpy_pickle.load(filename_raw)
        nose.tools.assert_equal(obj, obj_reloaded)

        # Do some cleanup
        os.remove(filename_raw)
        if os.path.exists(filename_compressed):
            os.remove(filename_compressed)


def test_compression_using_file_extension():
    # test that compression method corresponds to the given filename extension.
    extensions_dict = {
        # valid compressor extentions
        '.z': 'zlib',
        '.gz': 'gzip',
        '.bz2': 'bz2',
        '.lzma': 'lzma',
        '.xz': 'xz',
        # invalid compressor extensions
        '.pkl': 'not-compressed',
        '': 'not-compressed'
    }
    filename = env['filename'] + str(random.randint(0, 1000))
    obj = "object to dump"

    for ext, cmethod in extensions_dict.items():
        dump_fname = filename + ext
        if not PY3_OR_LATER and cmethod in ('xz', 'lzma'):
            # Lzma module only available for python >= 3.3
            msg = "{0} compression is only available".format(cmethod)
            assert_raises_regex(NotImplementedError, msg,
                                numpy_pickle.dump, obj, dump_fname)
        else:
            numpy_pickle.dump(obj, dump_fname)
            # Verify the file contains the right magic number
            with open(dump_fname, 'rb') as f:
                nose.tools.assert_equal(
                    _detect_compressor(f), cmethod)
            # Verify the reloaded object is correct
            obj_reloaded = numpy_pickle.load(dump_fname)
            nose.tools.assert_true(isinstance(obj_reloaded, type(obj)))
            nose.tools.assert_equal(obj_reloaded, obj)
            os.remove(dump_fname)


@with_numpy
def test_file_handle_persistence():
    objs = [np.random.random((10, 10)),
            "some data",
            np.matrix([0, 1, 2])]
    fobjs = [open]
    if not PY26:
        fobjs += [bz2.BZ2File, gzip.GzipFile]
    if PY3_OR_LATER:
        import lzma
        fobjs += [lzma.LZMAFile]
    filename = env['filename'] + str(random.randint(0, 1000))

    for obj in objs:
        for fobj in fobjs:
            with fobj(filename, 'wb') as f:
                numpy_pickle.dump(obj, f)

            # using the same decompressor prevents from internally
            # decompress again.
            with fobj(filename, 'rb') as f:
                obj_reloaded = numpy_pickle.load(f)

            # when needed, the correct decompressor should be used when
            # passing a raw file handle.
            with open(filename, 'rb') as f:
                obj_reloaded_2 = numpy_pickle.load(f)

            if isinstance(obj, np.ndarray):
                np.testing.assert_array_equal(obj_reloaded, obj)
                np.testing.assert_array_equal(obj_reloaded_2, obj)
            else:
                nose.tools.assert_equal(obj_reloaded, obj)
                nose.tools.assert_equal(obj_reloaded_2, obj)

            os.remove(filename)


@with_numpy
def test_in_memory_persistence():
    objs = [np.random.random((10, 10)),
            "some data",
            np.matrix([0, 1, 2])]
    for obj in objs:
        f = io.BytesIO()
        numpy_pickle.dump(obj, f)
        obj_reloaded = numpy_pickle.load(f)
        if isinstance(obj, np.ndarray):
            np.testing.assert_array_equal(obj_reloaded, obj)
        else:
            nose.tools.assert_equal(obj_reloaded, obj)


@with_numpy
def test_file_handle_persistence_mmap():
    obj = np.random.random((10, 10))
    filename = env['filename'] + str(random.randint(0, 1000))

    with open(filename, 'wb') as f:
        numpy_pickle.dump(obj, f)

    with open(filename, 'rb') as f:
        obj_reloaded = numpy_pickle.load(f, mmap_mode='r+')

    np.testing.assert_array_equal(obj_reloaded, obj)


@with_numpy
def test_file_handle_persistence_compressed_mmap():
    obj = np.random.random((10, 10))
    filename = env['filename'] + str(random.randint(0, 1000))

    with open(filename, 'wb') as f:
        numpy_pickle.dump(obj, f, compress=('gzip', 3))

    with closing(gzip.GzipFile(filename, 'rb')) as f:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            numpy_pickle.load(f, mmap_mode='r+')
            nose.tools.assert_equal(len(caught_warnings), 1)
            for warn in caught_warnings:
                nose.tools.assert_equal(warn.category, UserWarning)
                nose.tools.assert_equal(warn.message.args[0],
                                        'File "%(filename)s" is compressed '
                                        'using "%(compressor)s" which is not '
                                        'compatible with mmap_mode '
                                        '"%(mmap_mode)s" flag '
                                        'passed. mmap_mode option will be '
                                        'ignored.' %
                                        {'filename': "",
                                         'mmap_mode': 'r+',
                                         'compressor': 'GzipFile'})


@with_numpy
def test_file_handle_persistence_in_memory_mmap():
    obj = np.random.random((10, 10))
    buf = io.BytesIO()

    numpy_pickle.dump(obj, buf)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        numpy_pickle.load(buf, mmap_mode='r+')
        nose.tools.assert_equal(len(caught_warnings), 1)
        for warn in caught_warnings:
            nose.tools.assert_equal(warn.category, UserWarning)
            nose.tools.assert_equal(warn.message.args[0],
                                    'In memory persistence is not compatible '
                                    'with mmap_mode "%(mmap_mode)s" '
                                    'flag passed. mmap_mode option will be '
                                    'ignored.' % {'mmap_mode': 'r+'})


def test_binary_zlibfile():
    filename = env['filename'] + str(random.randint(0, 1000))

    # Test bad compression levels
    for bad_value in (-1, 10, 15, 'a', (), {}):
        nose.tools.assert_raises(ValueError,
                                 BinaryZlibFile, filename, 'wb',
                                 compresslevel=bad_value)

    # Test invalid modes
    for bad_mode in ('a', 'x', 'r', 'w', 1, 2):
        nose.tools.assert_raises(ValueError,
                                 BinaryZlibFile, filename, bad_mode)

    # Test wrong filename type (not a string or a file)
    for bad_file in (1, (), {}):
        nose.tools.assert_raises(TypeError,
                                 BinaryZlibFile, bad_file, 'rb')

    for d in (b'a little data as bytes.',
              # More bytes
              10000 * "{0}"
              .format(random.randint(0, 1000) * 1000).encode('latin-1')):
        # Regular cases
        for compress_level in (1, 3, 9):
            with open(filename, 'wb') as f:
                with BinaryZlibFile(f, 'wb',
                                    compresslevel=compress_level) as fz:
                    nose.tools.assert_true(fz.writable())
                    fz.write(d)
                    nose.tools.assert_equal(fz.fileno(), f.fileno())
                    nose.tools.assert_raises(io.UnsupportedOperation,
                                             fz._check_can_read)
                    nose.tools.assert_raises(io.UnsupportedOperation,
                                             fz._check_can_seek)
                nose.tools.assert_true(fz.closed)
                nose.tools.assert_raises(ValueError,
                                         fz._check_not_closed)

            with open(filename, 'rb') as f:
                with BinaryZlibFile(f) as fz:
                    nose.tools.assert_true(fz.readable())
                    if PY3_OR_LATER:
                        nose.tools.assert_true(fz.seekable())
                    nose.tools.assert_equal(fz.fileno(), f.fileno())
                    nose.tools.assert_equal(fz.read(), d)
                    nose.tools.assert_raises(io.UnsupportedOperation,
                                             fz._check_can_write)
                    if PY3_OR_LATER:
                        # io.BufferedIOBase doesn't have seekable() method in
                        # python 2
                        nose.tools.assert_true(fz.seekable())
                        fz.seek(0)
                        nose.tools.assert_equal(fz.tell(), 0)
                nose.tools.assert_true(fz.closed)

            os.remove(filename)

            # Test with a filename as input
            with BinaryZlibFile(filename, 'wb',
                                compresslevel=compress_level) as fz:
                nose.tools.assert_true(fz.writable())
                fz.write(d)

            with BinaryZlibFile(filename, 'rb') as fz:
                nose.tools.assert_equal(fz.read(), d)

            # Test without context manager
            fz = BinaryZlibFile(filename, 'wb', compresslevel=compress_level)
            nose.tools.assert_true(fz.writable())
            fz.write(d)
            fz.close()

            fz = BinaryZlibFile(filename, 'rb')
            nose.tools.assert_equal(fz.read(), d)
            fz.close()


###############################################################################
# Test dumping array subclasses
if np is not None:

    class SubArray(np.ndarray):

        def __reduce__(self):
            return _load_sub_array, (np.asarray(self), )

    def _load_sub_array(arr):
        d = SubArray(arr.shape)
        d[:] = arr
        return d

    class ComplexTestObject:
        """A complex object containing numpy arrays as attributes."""

        def __init__(self):
            self.array_float = np.arange(100, dtype='float64')
            self.array_int = np.ones(100, dtype='int32')
            self.array_obj = np.array(['a', 10, 20.0], dtype='object')


@with_numpy
def test_numpy_subclass():
    filename = env['filename']
    a = SubArray((10,))
    numpy_pickle.dump(a, filename)
    c = numpy_pickle.load(filename)
    nose.tools.assert_true(isinstance(c, SubArray))
    np.testing.assert_array_equal(c, a)


def test_pathlib():
    try:
        from pathlib import Path
    except ImportError:
        pass
    else:
        filename = env['filename']
        value = 123
        numpy_pickle.dump(value, Path(filename))
        nose.tools.assert_equal(numpy_pickle.load(filename), value)
        numpy_pickle.dump(value, filename)
        nose.tools.assert_equal(numpy_pickle.load(Path(filename)), value)


@with_numpy
def test_non_contiguous_array_pickling():
    filename = env['filename'] + str(random.randint(0, 1000))

    for array in [  # Array that triggers a contiguousness issue with nditer,
                    # see https://github.com/joblib/joblib/pull/352 and see
                    # https://github.com/joblib/joblib/pull/353
                    np.asfortranarray([[1, 2], [3, 4]])[1:],
                    # Non contiguous array with works fine with nditer
                    np.ones((10, 50, 20), order='F')[:, :1, :]]:
        nose.tools.assert_false(array.flags.c_contiguous)
        nose.tools.assert_false(array.flags.f_contiguous)
        numpy_pickle.dump(array, filename)
        array_reloaded = numpy_pickle.load(filename)
        np.testing.assert_array_equal(array_reloaded, array)
        os.remove(filename)


@with_numpy
def test_pickle_highest_protocol():
    # ensure persistence of a numpy array is valid even when using
    # the pickle HIGHEST_PROTOCOL.
    # see https://github.com/joblib/joblib/issues/362

    filename = env['filename'] + str(random.randint(0, 1000))
    test_array = np.zeros(10)

    numpy_pickle.dump(test_array, filename, protocol=pickle.HIGHEST_PROTOCOL)
    array_reloaded = numpy_pickle.load(filename)

    np.testing.assert_array_equal(array_reloaded, test_array)
