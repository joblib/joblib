""" Central data store to store information on the cache entries.
"""
import os
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
            os.fdopen(fd).write('0')
        except OSError:
            pass
        # The registry file
        self.registry_name = os.path.join(dir_name, 'registry')


    def increment_size(self, size):
        """ Increment the size stored on the disk.

            This is a global lock and possible bottleneck accross
            processes. Don't call it too often.
        """
        with LockedFile(self.size_file) as f:
            current_size = int(size_file.read())
            size_file.seek(0)
            total_size = size + current_size
            size_file.write('%i' % (total_size))
        return total_size


    def add_entry(self, func_name, argument_hash, module, computation_time, 
                  size, access_time, last_cost):
        """ Add a line to the registry.
        """
        file(self.registry_name, 'ab').write(
                '%s, %s, %s, %s, %s, %s, %s\n' % (
                    func_name, argument_hash, module, computation_time, 
                    size, access_time, last_cost))

