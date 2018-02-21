.PHONY: all test test-no-multiprocessing doc-html doc-clean

all: test

test:
	pytest joblib --timeout 15 -vl

test-no-multiprocessing:
	export JOBLIB_MULTIPROCESSING=0 && pytest joblib

# generate html documentation with warning as errors
doc-html:
	@cd doc && sphinx-build -W -b html . ../build/sphinx/html/

# remove generated sphinx gallery examples and sphinx documentation
doc-clean:
	@rm -rf doc/auto_examples && rm -rf doc/generated && rm -rf build
