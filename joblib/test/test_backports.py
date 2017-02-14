import mmap

from joblib.backports import make_memmap, concurrency_safe_rename
from joblib.test.common import with_numpy
from joblib.testing import parametrize


@with_numpy
def test_memmap(tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    memmap_obj = make_memmap(fname, shape=size, mode='w+', offset=offset)
    assert memmap_obj.offset == offset


@parametrize('src_content', [None, 'src content'])
def test_concurrency_safe_rename(tmpdir, src_content):
    src_path = tmpdir.join('src')
    src_path.write('src content')
    dst_path = tmpdir.join('dst')
    if src_content is not None:
        dst_path.write('dst content')

    concurrency_safe_rename(src_path.strpath, dst_path.strpath)
    assert not src_path.exists()
    assert dst_path.exists()
    assert dst_path.read() == 'src content'
