""" Central data store to store information on the cache entries.
"""
import os
import shutil
import errno
import pickle
import time
import itertools

from .locked_file import LockedFile

def cumulative_cost(last_cost, size, computation_time, access_times, 
                    current_time):
    """ Compute the cumulative cost acquired by the entry
    """
    end_times = list(access_times[1:])
    end_times.append(current_time)
    cost = last_cost
    for start, end in itertools.izip(access_times, end_times):
        delta_t = max(1e-6, end - start)
        alpha = 1 - computation_time/delta_t
        cost = alpha*cost + size
    return cost, start


def sort_entries(db):
    items = db.items()
    return sorted(items, key=lambda x: -x[1][0])

################################################################################
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
        self.compressing_registry = self.registry_name + '.compressing'
        self.registry_store = self.registry_name + '.store'
        try:
            fd = os.open(self.registry_store, os.O_EXCL | os.O_RDWR |
                                         os.O_CREAT)
            pickle.dump(dict(), os.fdopen(fd, 'w'))
        except OSError:
            pass


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

    
    def read_size(self):
        return int(open(self.size_file, 'r').read())


    def add_entry(self, module, func_name, argument_hash, computation_time, 
                  size, access_time, last_cost):
        """ Add a line to the registry.
        """
        file(self.current_registry, 'ab').write(
                '%s, %s, %s, %s, %s, %s, %s\n' % (
                    func_name, argument_hash, module, computation_time, 
                    size, access_time, last_cost))


    def compress(self):
        """ Compute costs for the current registry and merge them with
            the backing store.
        """
        # First thing, move the current registry, this creates a lock so
        # that other processes don't try to compress at the same time
        if os.path.exists(self.compressing_registry):
            # XXX: this is not a lock, and we have a race
            # condition here, maybe we should try opening a file with the
            # right flags to be exclusive
            # XXX: Should check how old the file is, to break the lock
            return
        try:
            os.rename(self.current_registry, self.compressing_registry)
        except IOError:
            # Oops, someone if modifying this file while we are
            return

        # Maybe we should use a locked file? XXX: Risk of deadlock
        with LockedFile(self.registry_store) as store_file:
            # First merge the two registries
            registry_file = open(self.compressing_registry, 'rb')
            db = pickle.load(store_file)
            for line in registry_file:
                # XXX: This should probably be in a separate function
                (func_name, argument_hash, module, computation_time, 
                                size, access_time, last_cost) = line.split(', ')
                last_cost = float(last_cost)
                computation_time = float(computation_time)
                size = float(size)
                access_time = float(access_time)
                key = '%s/%s:%s' % (module, func_name, argument_hash)
                if not key in db:
                    db[key] = [last_cost, size, computation_time, 
                                                    [access_time]]
                else:
                    db[key][-1].append(access_time)
            
            # Second update costs
            current_time = time.time()
            for key in db:
                last_cost, size, computation_time, access_times = db[key]
                access_times = sorted(access_times)
                cost, last_access = cumulative_cost(last_cost, size, 
                                            computation_time, 
                                            access_times, current_time)
                db[key] = cost, size, computation_time, [last_access]

            # Third remove entries
            db, size_gain = self._compress_and_flush(db)
            self.increment_size(-size_gain)
            store_file.seek(0)
            pickle.dump(db, store_file)


    def clear(self):
        # XXX: Will need to deal with locks
        # XXX: This is killing to much: our directory, our size_file...
        shutil.rmtree(self.dir_name)


    def _compress_and_flush(self, db, fraction=.1):
        """ Cache replacement: remove 'fraction' of the size of the stored 
            cache.
        """
        cachedir = self.dir_name
        cache_size = self.read_size()
        target_gain = (1-fraction) * cache_size
        size_gain = 0
        for db_entry in sort_entries(db):
            key, (cost, size, computation_time, access_times) = db_entry
            argument_dir = os.path.join(cachedir, key)
            if os.path.exists(argument_dir):
                self._rm_dir(argument_dir)
            db.remove(key)
            size_gain += size
            #if safe_listdir(func_dir) == ['func_code.py']:
            #    try:    
            #        shutil.rmtree(func_dir)
            #    except:
            #        # XXX: Where is our logging framework?
            #        print ('[joblib] Warning could not empty cache directory %s'
            #                % func_dir)
            if size_gain >= target_gain:
                break
        return db, size_gain


    def _rm_dir(self, dir_name):
        try:
            shutil.rmtree(dir_name)
        except:
            # XXX: Where is our logging framework?
            print (
            '[joblib] Warning could not empty cache directory %s'
                    % dir_name)


if __name__ == '__main__':
    import random
    def random_entry(hash=None, func_name=None, module=None):
        if hash is None:
            hash = hex(random.randint(0, 100))
        if func_name is None:
            func_name = str(random.random())
        if module is None:
            module = str(random.random())
        size = random.randint(1, 1000)
        access_time = time.time()
        computation_time = 10*random.random()
        return (func_name, hash, module, computation_time,
                            size, access_time, float(size))

    registry = Registry('./jobcache')
    registry.clear()
    registry = Registry('./jobcache')
    registry.increment_size(10)
    for _ in range(10):
        this_entry = random_entry()
        func_name = this_entry[0]
        hash = this_entry[1]
        module = this_entry[2]
        for _ in range(5):
            registry.add_entry(*random_entry(hash=hash,
                               func_name=func_name, module=module))

    registry.compress()

