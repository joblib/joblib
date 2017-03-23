

all: test

test:
	pytest joblib

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && pytest joblib
