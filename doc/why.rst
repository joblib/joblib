
Why joblib: project goals
===========================

What pipelines bring us
--------------------------

Pipeline processing systems can provide a set of useful features:

Data-flow programming for performance
......................................

* **On-demand computing:** in pipeline systems such as labView, or VTK
  calculations are performed as needed by the outputs and only when
  inputs change.

* **Transparent parallelization:** a pipeline topology can be inspected
  to deduce which operations can be run in parallel (it is equivalent to
  purely functional programming).

Provenance tracking for understanding the code
...............................................

* **Tracking of data and computations:** to be able to fully reproduce a
  computational experiment: requires tracking of the data and operation
  implemented.

* **Inspecting data flow:** Inspecting intermediate results helps
  debugging and understanding.

.. topic:: But pipeline frameworks can get in the way
    :class: warning

    We want our code to look like the underlying algorithm,
    not like a software framework.

Joblib's approach
--------------------

Functions are the simplest abstraction used by everyone. Our pipeline
jobs (or tasks) are made of decorated functions.

Tracking of parameters in a meaningful way requires specification of
data model. We give up on that and use hashing for performance and
robustness.

Design choices
---------------

* No dependencies other than Python

* Robust, well-tested code, at the cost of functionality

* Fast and suitable for scientific computing on big dataset without
  changing the original code

* Only local imports: **embed joblib in your code by copying it**



