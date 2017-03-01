

all: test

test:

	pytest joblib --timeout 15

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && pytest joblib
