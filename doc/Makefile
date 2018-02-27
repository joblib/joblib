.PHONY: all doc clean

all: doc

# generate html documentation with warning as errors
doc:
	sphinx-build -W -b html . ./_build/sphinx/html/

# remove generated sphinx gallery examples and sphinx documentation
clean:
	rm -rf auto_examples && rm -rf generated && rm -rf _build
