

all: test

test:
	pytest -v joblib

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && pytest joblib
