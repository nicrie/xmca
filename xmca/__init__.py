import os

version_path = os.path.dirname(os.path.dirname(__file__))
version_file = open(os.path.join(version_path, 'version'))
version = version_file.read().rstrip('\n')

__version__ = version
__author__ = 'Niclas Rieger'
__email__ = 'niclasrieger@gmail.com'
