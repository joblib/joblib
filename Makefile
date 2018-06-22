.PHONY: all test test-no-multiprocessing test-doc doc doc-clean

all: test

test:
	pytest joblib --timeout 30 -vl

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && pytest joblib

test-doc:
	pytest $(shell find doc -name '*.rst' | sort)

# generate html documentation using sphinx
doc:
	make -C doc

# clean documentation
doc-clean:
	make -C doc clean
