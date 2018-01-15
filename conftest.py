from distutils.version import LooseVersion

import pytest
from _pytest.doctest import DoctestItem


def pytest_collection_modifyitems(config, items):
    # numpy changed the str/repr formatting of numpy arrays in 1.14. We want to
    # run doctests only for numpy >= 1.14.
    try:
        import numpy as np
    except ImportError:
        return

    if LooseVersion(np.__version__) >= LooseVersion('1.14'):
        return

    skip_marker = pytest.mark.skip(
        reason='doctests are only run for numpy >= 1.14')

    for item in items:
        if isinstance(item, DoctestItem):
            item.add_marker(skip_marker)
