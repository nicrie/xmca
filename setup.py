import os

from setuptools import find_packages, setup
'''
Run the following code in your conda environment to make the package available
in development mode
$ python setup.py develop
'''

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    install_requires = fh.read()

requirements = ['numpy >= 1.19.2',
                'xarray >= 0.16.2',
                'matplotlib >= 3.3.2',
                'statsmodels >= 0.12.2',
                'tqdm',
                'cartopy >= 0.18.0']

# Get version file
here = os.path.dirname(__file__)
version_file = os.path.join(here, 'version')

setup(
    name='xmca',
    include_package_data=True,
    keywords='eof, analysis, mca, pca',
    author='Niclas Rieger',
    author_email='niclasrieger@gmail.com',
    description='Maximum Covariance Analysis in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nicrie/xmca',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    version_config={
        'vesion_file' : version_file,
        'dev_template': '{tag}.post{ccount}',
        'dirty_template': '{tag}.post{ccount}'
    },
    setup_requires=['setuptools-git-versioning', 'numpy'],
    install_requires=requirements,
)
