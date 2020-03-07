def test_data(arr):
    return arr[0:20]


def crash(arr):
    from joblib._memmapping_reducer import _get_backing_memmap
    print(_get_backing_memmap(arr).filename)
    raise ValueError
