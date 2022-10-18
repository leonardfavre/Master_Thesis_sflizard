"""Copyright (C) SquareFactory SA - All Rights Reserved.

This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
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
