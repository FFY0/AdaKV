from setuptools import setup, find_packages
NAME = 'adakv'
setup(
    name=NAME,
    version="0.0.1",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"])
)
