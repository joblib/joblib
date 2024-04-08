.. raw:: html

  <style type="text/css">
    div.body li.toctree-l1 {
        padding: 0.5em 0 1em 0 ;
        list-style-type: none;
        font-size: 150% ;
        }

    div.body li.toctree-l2 {
        font-size: 70% ;
        list-style-type: square;
        }

    div.body li.toctree-l3 {
        font-size: 85% ;
        list-style-type: circle;
        }

    div.bodywrapper blockquote {
	margin: 0 ;
    }

  </style>


Joblib: running Python functions as pipeline jobs
=================================================

Introduction
------------


.. automodule:: joblib

 .. toctree::
    :maxdepth: 2
    :caption: User manual

    why.rst
    installing.rst
    memory.rst
    parallel.rst
    persistence.rst
    auto_examples/index
    developing.rst

.. currentmodule:: joblib

Module reference
----------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :caption: Module reference

   Memory
   Parallel
   parallel_config

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dump
   load
   hash
   register_compressor

Deprecated functionalities
--------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   :caption: Deprecated functionalities

   parallel_backend