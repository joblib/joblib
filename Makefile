

all: test

test:
	py.test joblib

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && py.test joblib
