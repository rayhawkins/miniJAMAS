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

import copy
import os

# GPU initialization (Windows only).
# Needs to be up here to load before tensorflow is imported through the pyjamas imports.
if os.name == 'nt':
    try:
        os.add_dll_directory(os.environ['CUDA_PATH'] + "\\bin")
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    except:
        print("CUDA not found, no GPU will be used.")
    else:
        print(f"CUDA found at {os.environ['CUDA_PATH']}" + "\\bin" + ", GPU initialized.")
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"] = "1"  # gets around compatibility issues with openssl when executing the app.

import sys
from typing import List, Tuple, Optional

import numpy
from skimage.measure import points_in_poly

from pyjamas.rannotations.rpolyline import RPolyline
import pyjamas.rimage.rimcore as rimcore
from pyjamas.rimage.rimml.batchml import BatchML
from pyjamas.rimage.rimml.batchclassifier import BatchClassifier
from pyjamas.rimage.rimml.batchneuralnet import BatchNeuralNet
import pyjamas.rimage.rimml.rimlr as rimlr
import pyjamas.rimage.rimml.rimsvm as rimsvm
import pyjamas.rimage.rimml.rimunet as rimunet


class PyJAMAS:
    """
    PyJAMAS is Just A More Awesome Siesta.

    Uses calendar versioning (https://calver.org).
    Format: YYYY.M.minor

    YYYY - Full year - 2006, 2016, 2106
    M - Short month - 6, 16, 10
    minor - minor version number, starts at 0 for a given YYYY.M combination.


    PyJAMAS() creates a PyJAMAS user interface.

    """

    folder = os.path.split(__file__)[0]

    # Data file extension.
    data_extension: str = '.pjs'
    matlab_extension: str = '.mat'
    image_extensions: Tuple[str] = rimcore.rimage.image_extensions
    plugin_extension: str = '.py'
    notebook_extension: str = '.ipynb'
    classifier_extension: str = '.cfr'
    backgroundimage_extension: str = '.bg'

    # Read version.
    __version__: str = '2024.12.0'

    def __init__(self):
        self.brush_size: int = 3  # brush size to paint polylines
        self.margin_size: int = 0  # margin size for cropping
        self.crop_tracked_polyline: bool = False  # crop function takes polyline on one slice or all slices
        self.fps: int = 7  # frames per second used to play the current sequence or when exporting as a movie
        self.close_all_polylines: bool = False  # close all polylines loaded from file or load as open polylines those whose first and last point are not the same
        self.batch_classifier: BatchML = None
        self.gaussian_sigma: float = 0.0
        self.scale_factor: Tuple[float, float] = (1.0, 1.0)

        self.min_pix_percentile: int = 0  # Lowest percentile of the pixel values to map to display value 0.
        self.max_pix_percentile: int = 100  # Highest percentile of the pixel values to map to display value 255.

        self.cwd = os.getcwd()
        self.pjs_path, _ = os.path.split(os.path.realpath(__file__))

        self.plugin_list: List = []
        self.plugin_path_list: List = []
        self.plugin_path: str = os.path.join(self.pjs_path, 'plugins')
        sys.path.append(self.plugin_path)

        self._copied_poly_ = None  # Stores copied polyline to be pasted.
        self._agraphicsitem_ = None  # Stores a graphicsitem transiently (e.g. a rectangle as it is being dragged).
        self.slicetracker: tuple = None

        self.slices = None
        self.curslice: int = None
        self.n_frames: int = None
        self.height: int = None  # number of rows.
        self.width: int = None  # number of columns.
        self.min_pix_percentile = 0
        self.max_pix_percentile = 100

        self.fiducials: list = None
        self.polylines: list = None
        self.polyline_ids: list = None
        self.vectors: list = None

        return True

    def initImage(self):
        self.fiducials = [[] for _ in range(self.n_frames)]
        self.polylines = [[] for _ in range(self.n_frames)]
        self.polyline_ids = [[] for _ in range(self.n_frames)]
        self.vectors = [[] for _ in range(self.n_frames)]

        # Make sure to continue to store the classifier in memory.
        if self.batch_classifier is None:
            self.batch_classifier = BatchClassifier(self.n_frames)
        elif type(self.batch_classifier.image_classifier) in [rimlr.lr, rimsvm.svm]:
            self.batch_classifier = BatchClassifier(self.n_frames, self.batch_classifier.image_classifier)
        elif type(self.batch_classifier.image_classifier) is rimunet.UNet:
            self.batch_classifier = BatchNeuralNet(self.n_frames, self.batch_classifier.image_classifier)

    def addFiducial(self, x, y, z):
        self.fiducials[z].append([x, y])

        return True

    def findGraphicItem(self, x: int, y: int, radius: int):
        """TODO: reimplement this function without using the PyQT6 GUI."""
        return None

    def find_clicked_polyline(self, x: int, y: int) -> int:
        """TODO: reimplement this function without using the PyQT6 GUI."""
        return None

    def removeFiducial(self, x: int, y: int, z: int):
        deleteCoords = [x, y, z]
        if deleteCoords in self.fiducials[z]:
            self.fiducials[z].remove(deleteCoords)

        return True

    def removeFiducialsPolyline(self, polyline: RPolyline = None, inside_flag: bool = True, z: int = None):
        # Go through the list of fiducials.
        if z is None:
            z = self.curslice

        polyline_list = polyline.points
        inside_poly_flags: numpy.ndarray = points_in_poly(self.fiducials[z], polyline_list)

        # To avoid deleting fiducials from the list we are checking.
        fiducial_list = self.fiducials[z].copy()

        for thefiducial, is_inside in zip(fiducial_list, inside_poly_flags):
            if (is_inside and inside_flag) or not (is_inside or inside_flag):
                self.removeFiducial(thefiducial[0], thefiducial[1], z)

        return True

    def addPolyline(self, coordinates: list, z: int, theid: int = None) -> bool:
        thepolyline = RPolyline(coordinates)

        self.polylines[z].append(thepolyline)

        if theid is not None and type(theid) == int and theid > 0:
            polyline_id = theid
        else:
            polyline_id = max(self.polyline_ids[z]) + 1 if len(self.polyline_ids[z]) else 1
        self.polyline_ids[z].append(polyline_id)

        return True

    def replacePolyline(self, index: int, coordinates: list, z: int) -> bool:
        thepolyline = RPolyline(coordinates)
        self.polylines[z][index] = thepolyline

        return True

    def removePolyline(self, x: int, y: int, z: int) -> bool:
        """TODO: reimplement without using the PyQt6 functions."""
        index_polyline = None
        poly = self.polylines[z].pop(index_polyline)
        poly_id = self.polyline_ids[z].pop(index_polyline)
        return True

    def removePolylineByIndex(self, index: int, z: int):
        poly = self.polylines[z].pop(index)
        poly_id = self.polyline_ids[z].pop(index)
        return poly, poly_id

    @classmethod
    def new_pjs(cls, theimage: numpy.ndarray):
        thenewpjs: PyJAMAS = PyJAMAS()
        thenewpjs.io.cbLoadArray(theimage)

        return thenewpjs

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self, memodict={}):
        newone = type(self)()
        memodict[id(self)] = newone

        # Deep copy methods with a __deepcopy__ magic method.
        # Otherwise, copy by reference.
        for k, v in self.__dict__.items():
            if getattr(v, "__deepcopy__", None):
                setattr(newone, k, v.__deepcopy__(memodict))
            else:
                setattr(newone, k, v)

        return newone

    def __str__(self) -> str:
        the_string: str = f"file name: {self.filename}\n" \
                          f"size (width, height, slices): ({self.width}, {self.height}, {self.n_frames})\n" \
                          f"display percentiles (min, max): ({self.min_pix_percentile}, {self.max_pix_percentile})\n" \
                          f"brush size (pixels): {self.brush_size}\n" \
                          f"current slice: {self.curslice + 1}\n" \
                          f"\tnumber of fiducials: {len(self.fiducials[self.curslice])}\n" \
                          f"\tnumber of polylines: {len(self.polylines[self.curslice])}"

        return the_string

    def copy_annotations(self, arange: Optional[Tuple[int, int]] = None) -> Tuple:
        if not arange:
            arange = (0, self.n_frames)

        return (copy.deepcopy(self.fiducials[arange[0]:arange[1]]),
                [[RPolyline(apolyline) for apolyline in atimepoint]
                 for atimepoint in self.polylines[arange[0]:arange[1]]],
                copy.deepcopy(self.polyline_ids[arange[0]:arange[1]]))

    def paste_annotations(self, theannotations: Tuple, arange: Optional[Tuple[int, int]] = None) -> bool:
        if len(theannotations) == 3:
            fiducials, polylines, polyline_ids = theannotations
        else:
            return False

        if not arange:
            arange = (0, self.n_frames)

        self.fiducials[arange[0]:arange[1]] = fiducials
        self.polylines[arange[0]:arange[1]] = polylines
        self.polyline_ids[arange[0]:arange[1]] = polyline_ids

        return True
