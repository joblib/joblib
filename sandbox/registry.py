""" Central data store to store information on the cache entries.
"""
import os
import shutil
import errno

from locked_file import LockedFile

class Registry(object):

    def __init__(self, dir_name):
        self.dir_name = dir_name
        # Create the directory
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if not e.errno == errno.EEXIST:
                raise e
        # Create the file
        self.size_file = os.path.join(dir_name, 'cache_size')
        try:
            fd = os.open(self.size_file, os.O_EXCL | os.O_RDWR |
                                         os.O_CREAT)
            os.fdopen(fd, 'w').write('0')
        except OSError:
            pass
        # The registry file
        self.registry_name    = os.path.join(dir_name, 'registry')
        self.current_registry = self.registry_name + '.current'


    def increment_size(self, size):
        """ Increment the size stored on the disk.

            This is a global lock and possible bottleneck accross
            processes. Don't call it too often.
        """
        with LockedFile(self.size_file) as f:
            current_size = int(f.read())
            f.seek(0)
            total_size = size + current_size
            f.write('%i' % (total_size))
        return total_size


    def add_entry(self, module, func_name, argument_hash, computation_time, 
                  size, access_time, last_cost):
        """ Add a line to the registry.
        """
        file(self.current_registry, 'ab').write(
                '%s, %s, %s, %s, %s, %s, %s\n' % (
                    func_name, argument_hash, module, computation_time, 
                    size, access_time, last_cost))


    def clear(self):
        # XXX: Will need to deal with locks
        # XXX: This is killing to much: our directory, our size_file...
        shutil.rmtree(self.dir_name)

if __name__ == '__main__':
    import random
    registry = Registry('./jobcache')
    registry.clear()
    registry = Registry('./jobcache')
    registry.increment_size(10)
    for _ in range(10):
        hash = hex(random.randint(0, 100))
        for _ in range(5):
            registry.add_entry('math', 'sqrt', hash, 
                random.randint(0, 100), 
                random.randint(0, 100), 
                random.randint(0, 100), 
                random.randint(0, 100))

