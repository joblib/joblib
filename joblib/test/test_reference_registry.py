
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.


from joblib.reference_registry import ReferenceRegistry

class AClass(object):
    """ A class for our tests. """


def test_registry_garbage_collection():
    """ Check that objects are cleaned from the registry when they are
        removed.
    """
    registry = ReferenceRegistry()
    a = AClass()
    b = AClass()
    registry.register(a)
    assert a in registry
    registry.register(b)
    assert b in registry
    del b
    assert len(registry.id_table.keys()) == 2
    del a
    assert len(registry.id_table.keys()) == 1

def test_registry_time_stamps():
    """ Check that time_stamps are well kept and ordered in the registry. 
    """
    registry = ReferenceRegistry()
    a = AClass()
    b = AClass()
    registry.register(a)
    registry.register(b)
    assert registry.latest_reference().id == id(b)

if __name__ == '__main__':
    test_registry_garbage_collection()
    test_registry_time_stamps()

