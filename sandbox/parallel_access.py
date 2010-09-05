"""
Some toy code to test parallel access of a file
"""
import time
import random
import os

from locked_file import LockedFile
from joblib import Parallel, delayed

central_file = 'central.db'

def append(i):
    random.seed()
    file('test.db', 'ab').write('% 3i %s\n' % (i, os.getpid()))
    with LockedFile(central_file) as size_file:
        current_value = int(size_file.read())
        size_file.seek(0)
        size_file.write('%i' % (i + current_value))
    time.sleep(.1*random.random())

if __name__ == '__main__':
    # having the file importable is needed to run on windows.
    file(central_file, 'wb').write('0')

    t0 = time.time()
    Parallel(n_jobs=20)(delayed(append)(i) 
                        for i in range(1000))
    print time.time() - t0

