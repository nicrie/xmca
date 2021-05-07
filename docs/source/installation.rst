Installation
============
The package can be installed via::

	pip install xmca

Dependencies
------------
The file ``requirements.txt`` lists all the dependencies. For
automatic installation, you may want to clone and run::

	pip install -r requirements.txt


Known Issues
------------
The dependencies of `cartopy`_ themselves are not installed via `pip` which is
why the setup will fail if some dependencies are not met. In this case, please
install ``cartopy`` first before installing ``xmca``.

Testing
-------
After cloning the repository::

	python -m unittest discover -v -s tests/


.. _cartopy: https://scitools.org.uk/cartopy/docs/latest/installing.html
