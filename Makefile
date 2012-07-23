

all: test 

test:
	nosetests

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && nosetests

