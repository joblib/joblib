

all: test 

test:
	nosetests joblib

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && nosetests

