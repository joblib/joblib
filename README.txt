Development
===================

Using bzr and launchpad
------------------------

See the following article, and the related ShowMeDo video:
http://wiki.showmedo.com/index.php/Using_Launchpad_and_Bazaar

Running the test suite
------------------------

To run the test suite, you need nosetests and the coverage modules.
Run the test suite using::

    nosetests

from the root of the project.


Building the docs
----------------------

To build the docs you need to have setuptools and sphinx (>=0.5) installed. 
Run the command::

    python setup.py build_sphinx

The docs are built in the build/sphinx/html directory.


Making a source tarball
----------------------------

To create a source tarball, eg for packaging or distributing, run the
following command::

    python setup.py sdist

The tarball will be created in the `dist` directory. This command will
compile the docs, and the resulting tarball can be installed with
no extra dependencies than the Python standard library. You will need
setuptool and sphinx.

Making a release and uploading it to PyPI
------------------------------------------

This command is only run by project manager, to make a release, and
upload in to PyPI::

    python setup.py sdist bdist_egg register upload

