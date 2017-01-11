

all: test

test:
	py.test joblib

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && py.test joblib

test-coverage:
	py.test --cov-config .coveragerc --cov-report \
		term-missing --cov=joblib joblib
