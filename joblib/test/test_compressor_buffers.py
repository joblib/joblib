import array
import io
import zlib

import pytest

from joblib.compressor import BinaryGzipFile, BinaryZlibFile


@pytest.mark.parametrize(
    "compressor,wbits", [(BinaryZlibFile, 15), (BinaryGzipFile, 31)]
)
@pytest.mark.parametrize(
    "make_data",
    [
        lambda: array.array("i", [1, 2, 3]),
        lambda: pytest.importorskip("numpy").arange(6, dtype="int64").reshape(2, 3),
        lambda: pytest.importorskip("numpy").array(42, dtype="int64"),
        lambda: memoryview(array.array("d", [1.5, 2.5])),
        lambda: b"plain bytes",
    ],
)
def test_compressor_counts_buffer_bytes(compressor, wbits, make_data):
    data = make_data()
    expected = memoryview(data).tobytes()
    destination = io.BytesIO()
    with compressor(destination, "wb") as writer:
        assert writer.write(data) == len(expected)
        assert writer.tell() == len(expected)
        assert writer.write(data) == len(expected)
        assert writer.tell() == 2 * len(expected)
    assert zlib.decompress(destination.getvalue(), wbits) == expected * 2
