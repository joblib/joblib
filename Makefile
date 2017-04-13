

all: test

test:

	pytest joblib --timeout 15 -vl

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && pytest joblib
