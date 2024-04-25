Installing joblib
===================

Using `pip`
------------

You can use `pip` to install joblib:

* For installing for all users, you need to run::

    pip install joblib

  You may need to run the above command as administrator

  On a unix environment, it is better to install outside of the hierarchy
  managed by the system::

    pip install --prefix /usr/local joblib

* Installing only for a specific user is easy if you use Python 2.7 or
  above::

    pip install --user joblib

Using distributions
--------------------

Joblib is packaged for several linux distribution: archlinux, debian,
ubuntu, altlinux, and fedora. For minimum administration overhead, using the
package manager is the recommended installation strategy on these
systems.

The manual way
---------------

To install joblib first download the latest tarball (follow the link on
the bottom of https://pypi.org/project/joblib/) and expand it.

Installing in a local environment
..................................

If you don't need to install for all users, we strongly suggest that you
create a local environment and install `joblib` in it. One of the pros of
this method is that you never have to become administrator, and thus all
the changes are local to your account and easy to clean up.
Simply move to the directory created by expanding the `joblib` tarball
and run the following command::

    python setup.py install --user

Installing for all users
........................

If you have administrator rights and want to install for all users, all
you need to do is to go in directory created by expanding the `joblib`
tarball and run the following line::

    python setup.py install

If you are under Unix, we suggest that you install in '/usr/local' in
order not to interfere with your system::

    python setup.py install --prefix /usr/local
