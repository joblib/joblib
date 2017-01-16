import mmap

from joblib.backports import make_memmap
from joblib.test.common import with_numpy


@with_numpy
def test_memmap(tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    memmap_obj = make_memmap(fname, shape=size, mode='w+', offset=offset)
    assert memmap_obj.offset == offset
