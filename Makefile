

all: test

test:
	py.test joblib --timeout 15

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && py.test joblib
