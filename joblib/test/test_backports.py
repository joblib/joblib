import mmap
import os

import pytest

from joblib import Parallel, delayed
from joblib.backports import concurrency_safe_rename, make_memmap
from joblib.test.common import with_numpy
from joblib.testing import parametrize


@with_numpy
def test_memmap(tmpdir):
    fname = tmpdir.join("test.mmap").strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    memmap_obj = make_memmap(fname, shape=size, mode="w+", offset=offset)
    assert memmap_obj.offset == offset


@parametrize("dst_content", [None, "dst content"])
@parametrize("backend", [None, "threading"])
def test_concurrency_safe_rename(tmpdir, dst_content, backend):
    src_paths = [tmpdir.join("src_%d" % i) for i in range(4)]
    for src_path in src_paths:
        src_path.write("src content")
    dst_path = tmpdir.join("dst")
    if dst_content is not None:
        dst_path.write(dst_content)

    Parallel(n_jobs=4, backend=backend)(
        delayed(concurrency_safe_rename)(src_path.strpath, dst_path.strpath)
        for src_path in src_paths
    )
    assert dst_path.exists()
    assert dst_path.read() == "src content"
    for src_path in src_paths:
        assert not src_path.exists()


@pytest.mark.skipif(os.name != "nt", reason="the retry loop only exists on Windows")
def test_concurrency_safe_rename_surfaces_the_last_error(monkeypatch):
    """The retry window expiring must report why the rename kept failing."""
    from joblib import backports

    denied = PermissionError("Access is denied")
    denied.winerror = 5

    def always_denied(src, dst):
        raise denied

    monkeypatch.setattr(backports, "replace", always_denied)

    with pytest.raises(PermissionError) as excinfo:
        backports.concurrency_safe_rename("src", "dst")

    # Previously this was a bare `raise` outside the except block, which gave
    # "RuntimeError: No active exception to reraise" instead.
    assert excinfo.value is denied
