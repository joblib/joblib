Installing joblib
===================

The `easy_install` way
-----------------------

For the easiest way to install joblib you need to have `setuptools`
installed.

* For installing for all users, you need to run::

    easy_install joblib

  You may need to run the above command as administrator

  On a unix environment, it is better to install outside of the hierarchy
  managed by the system::

    easy_install --prefix /usr/local joblib

* Installing only for a specific user is easy if you use Python 2.6 or
  above::

    easy_install --user joblib

.. warning::

    Packages installed via `easy_install` override the Python module look
    up mechanism and thus can confused people not familiar with
    setuptools. Although it may seem harder, we suggest that you use the
    manual way, as described in the following paragraph.

Using distributions
--------------------

Joblib is packaged for several linux distribution: archlinux, debian,
ubuntu, and altlinux. For minimum administration overhead, using the
package manager is the recommended installation strategy on these
systems.

The manual way
---------------

To install joblib first download the latest tarball (follow the link on
the bottom of http://pypi.python.org/pypi/joblib) and expand it.

Installing in a local environment
..................................

If you don't need to install for all users, we strongly suggest that you
create a local environment and install `joblib` in it. One of the pros of
this method is that you never have to become administrator, and thus all
the changes are local to your account and easy to clean up.

* **If you are under Pyton 2.6 or above**
  
  Simple go in the directory created by expanding the `joblib` tarball
  and run the following command::

    python setup.py install --user

* **If you are under Python 2.5**

    #. First, create the following directory (where `~` is your home
       directory, or any directory that you want to use as a base for
       your local Python environment, and `X` is your Python version
       number, e.g. `2.6`)::

	~/usr/lib/pythonX/site-packages

    #. Second, make sure that you add this directory in your environment
       variable `PYTHONPATH`. Under window you can do this by editing
       your environment variables in the system parameters dialog. Under
       Unix you can add the following line to your `.bashrc` or any file
       source at login::

	export PYTHONPATH=$HOME/usr/lib/python2.6/site-packages:$PYTHONPATH

    #. In the directory created by expanding the `joblib` tarball, run the
       following command::

	python setup.py install --prefix ~/usr

       You should not be required to become administrator, if you have
       write access to the directory you are installing to.

Installing for all users
........................

If you have administrator rights and want to install for all users, all
you need to do is to go in directory created by expanding the `joblib`
tarball and run the following line::

    python setup.py install

If you are under Unix, we suggest that you install in '/usr/local' in
order not to interfere with your system::

    python setup.py install --prefix /usr/local
