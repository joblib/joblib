

all: test 

test:
	nosetests

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && nosetests

doc:
	python setup.py build_sphinx

.PHONY: all doc test test-no-multiprocessing