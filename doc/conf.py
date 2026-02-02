# -*- coding: utf-8 -*-
#
# The contents of this file are pickled, so don't put values in the
# namespace that aren't pickleable (module imports are okay,
# they're removed automatically).
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import os
import re
import sys
from datetime import datetime

import joblib

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
]

autosummary_generate = True

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
}

# sphinx-gallery configuration
sphinx_gallery_conf = {
    "default_thumb_file": "_static/joblib_logo_examples.png",
    "doc_module": "joblib",
    "filename_pattern": "",
    "ignore_pattern": "utils.py",
    "backreferences_dir": os.path.join("generated"),
    "reference_url": {"joblib": None},
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = {".rst": "restructuredtext"}

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "joblib"
year = datetime.now().year
copyright = f"2008-{year}, Joblib developers"

# The full version, including alpha/beta/rc tags.
release = joblib.__version__
# The short X.Y version.
version = re.sub(r"(\d+\.\d+).*", r"\1", release)

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Avoid '+DOCTEST...' comments in the docs
trim_doctest_flags = True

# Options for HTML output
# -----------------------

html_static_path = ["_static"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/joblib/joblib",
    "logo": {
        "image_light": "joblib_logo.svg",
        "image_dark": "joblib_logo_dark.svg",
    },
    "switcher": {
        "json_url": "https://joblib--1774.org.readthedocs.build/en/1774/_static/versions.json",
        "version_match": release
	},
    
    "navbar_start": [
        "navbar-logo",
        "version-switcher",
	],
    "navbar_end": [
        "theme-switcher",
        "navbar-icon-links",
    ],
    "footer_start": [],
    "footer_end": [
        "copyright",
    ],
}

html_favicon = "_static/favicon.ico"

html_css_files = [
    "custom.css",
    "custom_pygments.css",
]

htmlhelp_basename = "joblibdoc"

# Options for LaTeX output
# ------------------------

latex_documents = [
    ("index", "joblib.tex", "joblib Documentation", "Gael Varoquaux", "manual"),
]

##############################################################################
# Hack to copy the CHANGES.rst file
import shutil  # noqa: E402

try:
    shutil.copyfile("../CHANGES.rst", "CHANGES.rst")
    shutil.copyfile("../README.rst", "README.rst")
except IOError:
    pass
    # This fails during the testing, as the code is ran in a different
    # directory

numpydoc_show_class_members = False

suppress_warnings = ["image.nonlocal_uri"]
