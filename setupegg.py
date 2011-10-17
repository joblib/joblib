#!/usr/bin/env python
"""Wrapper to run setup.py using setuptools."""

import zipfile
import os
import sys

from setuptools import Command
from sphinx_pypi_upload import UploadDoc

###############################################################################
# Code to copy the sphinx-generated html docs in the distribution.
DOC_BUILD_DIR = os.path.join('build', 'sphinx', 'html')


def relative_path(filename):
    """ Return the relative path to the file, assuming the file is
        in the DOC_BUILD_DIR directory.
    """
    length = len(os.path.abspath(DOC_BUILD_DIR)) + 1
    return os.path.abspath(filename)[length:]


class ZipHelp(Command):
    description = "zip the help created by the build_sphinx, " + \
                  "and put it in the source distribution. "

    user_options = [
        ('None', None, 'this command has no options'),
        ]

    def run(self):
        if not os.path.exists(DOC_BUILD_DIR):
            raise OSError('Doc directory does not exist.')
        target_file = os.path.join('doc', 'documentation.zip')
        # ZIP_DEFLATED actually compresses the archive. However, there
        # will be a RuntimeError if zlib is not installed, so we check
        # for it. ZIP_STORED produces an uncompressed zip, but does not
        # require zlib.
        try:
            zf = zipfile.ZipFile(target_file, 'w',
                                            compression=zipfile.ZIP_DEFLATED)
        except RuntimeError:
            zf = zipfile.ZipFile(target_file, 'w',
                                            compression=zipfile.ZIP_STORED)

        for root, dirs, files in os.walk(DOC_BUILD_DIR):
            relative = relative_path(root)
            if not relative.startswith('.doctrees'):
                for f in files:
                    zf.write(os.path.join(root, f),
                            os.path.join(relative, f))
        zf.close()

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class GenerateHelp(Command):
    description = " Generate the autosummary files "

    user_options = [
        ('None', None, 'this command has no options'),
        ]

    def run(self):
        os.system( \
            "%s doc/sphinxext/autosummary_generate.py " % sys.executable + \
            "-o doc/generated/ doc/*.rst")

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


###############################################################################
# Call the setup.py script, injecting the setuptools-specific arguments.

extra_setuptools_args = dict(
                            tests_require=['nose', 'coverage'],
                            test_suite='nose.collector',
                            cmdclass={'zip_help': ZipHelp,
                                      'generate_help': GenerateHelp,
                                      'upload_help': UploadDoc},
                            zip_safe=False,
                            )


if __name__ == '__main__':
    execfile('setup.py', dict(__name__='__main__',
                          extra_setuptools_args=extra_setuptools_args))
