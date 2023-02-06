import re

from setuptools import find_packages, setup

with open("sflizard/__init__.py") as f:
    version = re.search(r"\d.\d.\d", f.read()).group(0)  # type: ignore

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sflizard",
    version=version,
    install_requires=requirements,
    packages=find_packages(),
)
