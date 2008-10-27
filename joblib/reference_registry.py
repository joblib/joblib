
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.


from weakref import ref
import time

class Reference(object):
    """ All the information we want to track about objects.
    """
    def __init__(self, obj, time_stamp, desc=''):
        self.type = type(obj)
        if hasattr(obj, '__class__'):
            self.obj_class = obj.__class__
        else:
            self.obj_class = None
        self.repr = repr(obj)
        self.time_stamp = time_stamp
        self.id = id(obj)
        self.desc = desc

    def __repr__(self):
        return 'Ref to %s %s, id:%i, time_stamp %f' % (
            self.repr, self.desc, self.id, self.time_stamp)


class ReferenceRegistry(object):
    """ A registry to keep weak references to objects, and trac
        information about them.
    """

    def __init__(self, default_time_stamp=None):
        # A table storing all the information for the objects that we
        # track, by id.
        default_ref = Reference(None, time_stamp=0)
        self.id_table = {id(None):default_ref}

        self.default_time_stamp = default_time_stamp


    def register(self, obj, time_stamp=None, desc=''):
        """ Add a reference to the object in the registry.
        """
        if time_stamp is None:
            if self.default_time_stamp is not None:
                time_stamp = self.default_time_stamp
            else:
                time_stamp = time.time()
        reference =  Reference(obj, time_stamp=time_stamp, desc=desc)

        obj_id = id(obj)
        def erase_ref(weak_ref):
            """ A callback for removing the object from our tables when it
                is garbage-collected.
            """
            #print "%s cleaned" % self.id_table[obj_id]
            self.id_table.pop(obj_id)
        #try:
        reference._weakref = ref(obj, erase_ref)
        #except TypeError:
        #    " Cannot create a weak-ref. "
        self.id_table[id(obj)] = reference

    def latest_reference(self):
        """ Returns the reference with the latest time stamp.
        """
        key = lambda ref: ref.time_stamp
        return max(self.id_table.values(), key=key)

    def update(self, registry):
        """ Merge the content of another registry in this one.
        """
        self.id_table.update(registry.id_table)

    def __contains__(self, obj):
        return (id(obj) in self.id_table)

    def __getitem__(self, obj):
        return self.id_table[id(obj)]

    def __iter__(self):
        return self.id_table.__iter__()

