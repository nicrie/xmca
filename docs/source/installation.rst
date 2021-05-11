Installation
------------

Installation is simply performed via

::

    pip install xmca

Known Issues
''''''''''''

Actually ``pip`` should take care of installing the correct
dependencies. However, the dependencies of ``cartopy`` itself are not
installed via ``pip`` which is why the setup may fail in some cases. If
so, please
`install <https://scitools.org.uk/cartopy/docs/latest/installing.html>`__
``cartopy`` first before installing ``xmca``. If you are using a
``conda`` environment, this can be achieved by

::

    conda install cartopy

Testing
'''''''

After cloning the repository

::

    python -m unittest discover -v -s tests/
