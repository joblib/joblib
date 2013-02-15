#!/usr/bin/env python
# -*- coding: utf-8 -*-:

import joblib.memory
import shutil



memory=joblib.memory.Memory(verbose=4, cachedir='testdepends')

def producer(X):
    """produce the big object"""
    print 'Call of producer', X
    return 10+X+1

joblib.memory.add_dependency(producer)
@memory.cache(ignore=['producee'], depends=['producer'])
def cached(producee, label):
    print 'Call cached', label
    return -producee1 



producee1 = producer(1)
producee2 = producer(2)


cached1 = cached(producee1, 'lab1')
cached2 = cached(producee2, 'lab2')
cached3 = cached(producee1, 'lab1')
cached3 = cached(producee1, 'lab1')
cached3 = cached(producee1, 'lab1')
cached3 = cached(producee1, 'lab1')



