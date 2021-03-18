import setuptools

"""
Run the following code in your conda environment to make the package available
$ python setup.py develop
"""


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "xmca",
    version =  "0.1.0",
    author = "Niclas Rieger",
    author_email = "niclasrieger@gmail.com",
    description = "Maximum Covariance Analysis in Python",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/NiclasRieger/xmca",
    packages = setuptools.find_packages(),
    license = "GPL-3.0",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
