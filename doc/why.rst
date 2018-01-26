
Why joblib: project goals
=========================

Benefits of pipelines
---------------------

Pipeline processing systems can provide a set of useful features:

Data-flow programming for performance
.....................................

* **On-demand computing:** in pipeline systems such as labView or VTK,
  calculations are performed as needed by the outputs and only when
  inputs change.

* **Transparent parallelization:** a pipeline topology can be inspected
  to deduce which operations can be run in parallel (it is equivalent to
  purely functional programming).

Provenance tracking to understand the code
..........................................

* **Tracking of data and computations:** This enables the reproducibility of a
  computational experiment.

* **Inspecting data flow:** Inspecting intermediate results helps
  debugging and understanding.

.. topic:: But pipeline frameworks can get in the way
    :class: warning

    Joblib's philosophy is to keep the underlying algorithm code unchanged,
    avoiding framework-style modifications.

Joblib's approach
-----------------

Functions are the simplest abstraction used by everyone. Pipeline
jobs (or tasks) in Joblib are made of decorated functions.

Tracking of parameters in a meaningful way requires specification of
data model. Joblib gives up on that and uses hashing for performance and
robustness.

Design choices
--------------

* No dependencies other than Python

* Robust, well-tested code, at the cost of functionality

* Fast and suitable for scientific computing on big dataset without
  changing the original code

* Only local imports: **embed joblib in your code by copying it**



