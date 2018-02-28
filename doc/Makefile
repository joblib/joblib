.PHONY: all html clean

all: html

# generate html documentation with warning as errors
html:
	sphinx-build -W -b html . ./_build/html/

# remove generated sphinx gallery examples and sphinx documentation
clean:
	rm -rf auto_examples && rm -rf generated && rm -rf _build
