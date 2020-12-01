# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='py3psdd',
    version='0.1', # get from pypsdd.__init__
    description='The Python PSDD Package, moved to python3',
    long_description=readme,
    author='Arthur Choi',
    author_email='aychoi@cs.ucla.edu',
    url='http://reasoning.cs.ucla.edu/psdd',
    python_requires='>=3.5',
    license=license,
    packages=["py3psdd"]
)
