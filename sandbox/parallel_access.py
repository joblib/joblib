"""
Some toy code to test parallel access of a file
"""
import time
import random
import os

from lockfile import FileLock
from joblib import Parallel, delayed

central_file = 'central.db'

def append(i):
    random.seed()
    file('test.db', 'ab').write('% 3i %s\n' % (i, os.getpid()))
    with FileLock(central_file, force=True, timeout=1):
    #try:
        current_value = int(file(central_file, 'rb').read())
        file(central_file, 'wb').write('%i' % (i + current_value))
    #except:
    #    pass
    time.sleep(.1*random.random())

file(central_file, 'wb').write('0')

t0 = time.time()
Parallel(n_jobs=20)(delayed(append)(i) 
                    for i in range(1000))
print time.time() - t0

