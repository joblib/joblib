

all: test 

test:
	py.test --pyargs joblib

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && py.test --pyargs joblib

