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

from typing import Optional

import numpy

from pyjamas.pjscore import PyJAMAS
from pyjamas.rcallbacks.rcallback import RCallback
from pyjamas.rutils import RUtils
from pyjamas.rimage.rimutils import rimutils
from pyjamas.rannotations.rpolyline import RPolyline

import skimage.morphology as skm


class RCBAnnotations(RCallback):
    PIX_SHIFT: int = 2  # x, y shift to use when pasting polylines with a shift with respect to original position.

    def cbCopyPolyline(self, theid: int, z: int = None) -> bool:
        """
        Copy the polyline at given z and index.
        :return: True.
        """
        if z is None:
            z = self.pjs.curslice
        theindex = self.pjs.polyline_ids[z].index(theid)
        self.pjs._copied_poly = self.pjs.polylines[z][theindex]
        return True

    def cbPastePolyline(self, z: int) -> object:
        """
        Paste the polyline previously copied (stored in pjs._copied_poly_).
        :return: True if polyline was copied, False otherwise.
        """

        if self.pjs._copied_poly_ is None or self.pjs._copied_poly_ == []:
            return False

        self.pjs.addPolyline(self.pjs._copied_poly_, self.pjs.curslice, pushundo=True)

        return True

    def cbMovePolyline(self, theid: int, z: int, dx: int, dy: int) -> bool:
        """
        Translate a specified polyline by a given increment in x (cols) and y (rows).
        :return: True.
        """
        theindex = self.pjs.polyline_ids[z].index(theid)
        self.pjs.polylines[z][theindex]

        return True

    def cbTrackFiducials(self, first: int = 0, last: int = None) -> bool:
        """
        Match fiducials across slices in a stack based on minimum distance. Requires a constant number of fiducials.
        :param first: start slice (>=0).
        :param last: final slice (<pjs.n_frames). Set to pjs.n_frames - 1 if not provided.
        :return: True if tracking finished correctly, False otherwise.
        """
        if last is None:
            last = self.pjs.n_frames - 1

        theslicenumbers = numpy.arange(first, last)
        self.track_fiducials(theslicenumbers)

        return True

    def track_fiducials(self, theslices: numpy.ndarray) -> bool:
        # Requires a constant number of fiducials.
        # Make sure that the slices are in a 1D numpy array.
        theslices = numpy.atleast_1d(theslices)
        num_slices = theslices.size

        # For every slice ...
        for i in range(num_slices-1):
            if len(self.pjs.fiducials[theslices[i]]) == 0:
                print(f"Stopping at slice {theslices[i]+1}: there are no fiducials to track there!")
                return True

            if len(self.pjs.fiducials[theslices[i+1]]) == 0:
                print(f"Stopping at slice {theslices[i+1]+1}: there are no fiducials to track there!")
                return True

            if len(self.pjs.fiducials[theslices[i]]) != len(self.pjs.fiducials[theslices[i+1]]):
                print(f"Error: slices {theslices[i]+1} and {theslices[i + 1] + 1} have a different number of fiducials.")
                return False

            fiducials_orig = numpy.array(self.pjs.fiducials[theslices[i]])
            fiducials_dest = numpy.array(self.pjs.fiducials[theslices[i + 1]])

            # Find distance between all pairs of fiducials.
            distance_matrix: numpy.ndarray = RUtils.point2point_distances(fiducials_orig, fiducials_dest)
            sorted_indices: numpy.ndarray = distance_matrix.argsort()

            # For each fiducials_orig[ii] select the closest fiducials_dest[jj].
            closest_fiducial_index: numpy.ndarray = sorted_indices[:, 0]

            # Deal with cases in which more than one fiducial maps to another.
            counts: int = numpy.bincount(closest_fiducial_index)

            if any(counts > 1):
                conflictive_indeces = numpy.arange(counts.shape[0])[counts > 1]
                original_fiducials = numpy.arange(closest_fiducial_index.shape[0])[closest_fiducial_index == conflictive_indeces[0]]
                print(f"Error in slice {theslices[i + 1]+1}, fiducial {conflictive_indeces[0]+1}: fiducials {original_fiducials+1} from slice {theslices[i]+1} map here.")
                return False

            else:
                self.pjs.fiducials[theslices[i+1]] = fiducials_dest[closest_fiducial_index].tolist()

            print(int((100*(i+1))/(num_slices-1)))

        return True

    def cbDeleteAllAnn(self) -> bool:
        """
        Delete annotations from all image slices.

        :return: True.
        """
        self.pjs.fiducials = [[] for _ in range(self.pjs.n_frames)]
        self.pjs.polylines = [[] for _ in range(self.pjs.n_frames)]
        self.pjs.polyline_ids = [[] for _ in range(self.pjs.n_frames)]

        return True

    def cbDeleteSliceAnn(self, index: Optional[int] = None) -> bool:
        """
        Delete annotations from a specific slice.

        :param index: index of the slice in which annotations will be deleted (>=0). Defaults to the current slice (pjs.curslice).
        :return: True.
        """

        if index is None or index is False:
            index = self.pjs.curslice

        self.pjs.fiducials[index] = []
        self.pjs.polylines[index] = []
        self.pjs.polyline_ids[index] = []

        return True

    def cbDeleteSlicePoly(self, index: Optional[int] = None) -> bool:
        """
        Delete polylines from a specific slice.

        :param index: index of the slice in which polylines will be deleted (>=0). Defaults to the current slice (pjs.curslice).
        :return: True.
        """

        if index is None or index is False:
            index = self.pjs.curslice

        self.pjs.polylines[index] = []
        self.pjs.polyline_ids[index] = []

        return True

    def cbDeleteSliceFiducials(self, index: Optional[int] = None) -> bool:
        """
        Delete fiducials from a specific slice.

        :param index: index of the slice in which fiducials will be deleted (>=0). Defaults to the current slice (pjs.curslice).
        :return: True
        """

        if index is None or index is False:
            index = self.pjs.curslice

        self.pjs.fiducials[index] = []

        return True

    def cbDeleteAllFiducials(self) -> bool:
        """
        Delete all fiducials from all slices.

        :return: True
        """

        self.pjs.fiducials = [[] for _ in range(self.pjs.n_frames)]

        return True

    def cbDeleteFiducialsOutsidePoly(self, theid: int, z: int = None) -> bool:
        """
        Remove fiducials outside the polyline of given id on given slice.
        :return: True.
        """
        if z is None:
            z = self.pjs.curslice

        theindex = self.pjs.polyline_ids[z].index(theid)
        thepolyline = self.pjs.polylines[z][theindex]
        self.pjs.removeFiducialsPolyline(thepolyline, False, z)
        return True

    def cbDeleteFiducialsInsidePoly(self, theid: int, z: int = None) -> bool:
        """
        Remove fiducials inside the polyline of given id on given slice.
        :return: True.
        """
        if z is None:
            z = self.pjs.curslice

        theindex = self.pjs.polyline_ids[z].index(theid)
        thepolyline = self.pjs.polylines[z][theindex]
        self.pjs.removeFiducialsPolyline(thepolyline, True, z)
        return True

    def cbDilatePolyline(self, theid: int, z: int, radius: int):
        theindex = self.pjs.polyline_ids[z].index(theid)
        thepolyline = self.pjs.polylines[z][theindex]
        obj_mask = rimutils.mask_from_polylines((self.pjs.height, self.pjs.width), [thepolyline], brushsz=0)
        obj_mask = skm.binary_dilation(obj_mask, skm.disk(radius))
        thepolylines = self.polylines_from_mask(obj_mask)

        return self.add_replace_polylines(thepolylines, theindex)

    def cbErodePolyline(self, theid: int, z: int, radius: int):
        theindex = self.pjs.polyline_ids[z].index(theid)
        thepolyline = self.pjs.polylines[z][theindex]
        obj_mask = rimutils.mask_from_polylines((self.pjs.height, self.pjs.width), [thepolyline], brushsz=0)
        obj_mask = skm.binary_erosion(obj_mask, skm.disk(radius))
        thepolylines = self.polylines_from_mask(obj_mask)

        return self.add_replace_polylines(thepolylines, theindex)
        return True

    @staticmethod
    def polylines_from_mask(obj_mask: numpy.ndarray) -> list:
        labelled_im, n_objs = skm.label(obj_mask, connectivity=1, background=0, return_num=True)
        contours = rimutils.extract_contours(labelled_im, border_objects=True)
        thepolylines = [RPolyline(numpy.asarray(this_contour)) for this_contour in contours]
        thepolylines[:] = [this_polyline if this_polyline.points[0] == this_polyline.points[-1]
                           else this_polyline.points << this_polyline.points[0]
                           for this_polyline in thepolylines]
        return thepolylines

    def add_replace_polylines(self, thepolylines: list, theindex: int) -> bool:
        # Remove polylines of 0 area
        areas = numpy.array([this_polyline.area() for this_polyline in thepolylines])
        thepolylines[:] = [thepolylines[index] for index in numpy.where(areas > 0)[0]]
        if len(thepolylines) > 0:  # Substitute the first new polyline in the same index as the previous, append others
            self.pjs.replacePolyline(theindex, thepolylines[0].points, self.pjs.curslice)
            if len(thepolylines) > 1:
                for this_polyline in thepolylines[1:]:
                    self.pjs.addPolyline(this_polyline, self.pjs.curslice)
        else:  # No polylines found, simply remove the old one
            self.pjs.removePolylineByIndex(theindex, self.pjs.curslice)

        return True
