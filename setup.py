"""
    PyJAMAS is Just A More Awesome Siesta
    Copyright (C) 2018  Rodrigo Fernandez-Gonzalez (rodrigo.fernandez.gonzalez@utoronto.ca)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import setuptools
import sys

from pyjamas.pjscore import PyJAMAS


with open("README.md", "r") as fh:
    long_description: str = fh.read()

bin_path: str = os.path.dirname(sys.executable)
# The interpreter needs to be modified to use pythonw, not python (this is a windowed app!!!).
interpreter_name: str = 'pythonw'
shebang: str = '#!/usr/bin/env '
sys.executable = os.path.join(bin_path, interpreter_name)
name = "pyjamas-rfglab"
description = "PyJAMAS is Just A More Awesome SIESTA"

version = PyJAMAS.__version__

# Build pyjamas.
# changes in install_requires should also be updated in docs/conf.py
setuptools.setup(
    name=name,
    version=version,
    author="Rodrigo Fernandez-Gonzalez",
    author_email="rodrigo.fernandez.gonzalez@utoronto.ca",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/rfg_lab/pyjamas",
    packages=setuptools.find_packages(exclude=("*tests*",)),
    # this does not yet exclude tests, but it will at some point: https://github.com/pypa/setuptools/issues/3260
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["ipywidgets>=8.0.4",
                      "joblib>=1.2.0", "lxml>=4.9.2", "matplotlib>=3.7.1",
                      "nbformat>=5.7.0",
                      "networkx>=2.8.4",
                      "numba==0.57.1",  ## 0.59.1 fails with "Measure polylines" 2024/04/17
                      "numpy==1.23.5",  ## 1.24.3 has an error when exporting mat files.
                      "opencv-python-headless==4.6.0.66",  ## greater than this will not work on Big Sur 2024/04/17
                      "pandas==2.1.4",  # avoid an issue with string cells, which must be initialized (see https://stackoverflow.com/questions/77098113/solving-incompatible-dtype-warning-for-pandas-dataframe-when-setting-new-column)
                      "pyqt6==6.4.2", ## 6.4.2 is the last version that works in Windows 11 2024/04/27
                      "pyqt6-qt6==6.4.2", ## 6.4.2 is the last version that works in Windows 11 2024/04/27
                      "scikit-image>=0.21.0",
                      "scikit-learn>=1.2.2",
                      "scipy>=1.10.1", "seaborn>=0.13.0", "setuptools>=67.8.0",
                      "shapely>=2.0.1",
                      "tensorflow==2.13.0", ## 6.4.2 is the last version that works in Windows 11 2024/04/27
                     ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'pyjamas=pyjamas.pjscore:main'
        ]
    }
)
