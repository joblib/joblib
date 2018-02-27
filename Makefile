.PHONY: all test test-no-multiprocessing doc doc-clean

all: test

test:
	pytest joblib --timeout 15 -vl

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && pytest joblib

# generate html documentation using sphinx
doc:
	make -C doc

# clean documentation
doc-clean:
	make -C doc clean
