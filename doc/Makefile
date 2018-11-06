# You can set these variables from the command lines
SPHINXOPTS   ?= -j $(shell nproc)
SPHINXBUILD  ?= sphinx-build

.PHONY: all html clean

all: html

# generate html documentation with warning as errors
html:
	$(SPHINXBUILD) -W -b html . ./_build/html/

# remove generated sphinx gallery examples and sphinx documentation
clean:
	rm -rf auto_examples && rm -rf generated && rm -rf _build
