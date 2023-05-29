# You can set these variables from the command lines
SPHINXOPTS   ?= -j $(shell nproc) -nWT --keep-going
SPHINXBUILD  ?= sphinx-build

.PHONY: all html clean

all: html

# generate html documentation with warning as errors
html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html . ./_build/html/

# remove generated sphinx gallery examples and sphinx documentation
clean:
	rm -rf auto_examples && rm -rf generated && rm -rf _build

view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/index.html')"

show: view
