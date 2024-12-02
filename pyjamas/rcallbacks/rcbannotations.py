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
import copy
from PyQt6 import QtCore, QtGui, QtWidgets

import pyjamas.dialogs as dialogs
from pyjamas.pjscore import PyJAMAS
from pyjamas.pjsthreads import ThreadSignals
from pyjamas.rcallbacks.rcallback import RCallback
from pyjamas.rutils import RUtils
from pyjamas.rimage.rimutils import rimutils
from pyjamas.rannotations.rpolyline import RPolyline

import skimage.morphology as skm


class RCBAnnotations(RCallback):
    PIX_SHIFT: int = 2  # x, y shift to use when pasting polylines with a shift with respect to original position.

    def cbNoAnn(self) -> bool:
        """
        Reset annotation mode: no action upon mouse click.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.no_annotations
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.ArrowCursor)

        return True

    def cbHideAnn(self) -> bool:
        """
        Toggle annotations off/on the screen.

        :return: True.
        """

        if self.pjs.show_annotations:
            self.pjs.eraseAnnotations()
        else:
            self.pjs.paintAnnotations()

        self.pjs.show_annotations = not self.pjs.show_annotations

        return True

    def cbFiducials(self) -> bool:
        """
        Set annotation mode to add/remove fiducials.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.fiducial_annotations
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbRectangles(self) -> bool:
        """
        Set annotation mode to add/remove rectangles (a specific type of polyline).

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.rectangle_annotations
        self.pjs._poly_ = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbPolylines(self) -> bool:
        """
        Set annotation mode to add/remove polylines.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.polyline_annotations
        self.pjs._poly_ = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbLiveWire(self) -> bool:
        """
        Set annotation mode to add/remove polylines using the LiveWire algorithm for semi-automated object delineation.
        The LiveWire is implemented using an A* algorithm, with a Euclidean distance heuristic that is weighted by
        self.pjs.livewire_gaussian_sigma (a weight of 0.0 is the equivalent of Dijkstra's minimal cost path search).

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.livewire_annotations
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbInflateBalloon(self) -> bool:
        """
        Set annotation mode to add/remove polylines using active contours for semi-automated object delineation.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.inflate_balloon
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbCopyPolyline(self) -> bool:
        """
        Set annotation mode to copy the polyline clicked on (stored in pjs._copied_poly_).

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.copy_polyline
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbPastePolyline(self, paste_shifted: bool = False) -> object:
        """
        Paste the polyline previously copied (stored in pjs._copied_poly_).

        :return: True if polyline was copied, False otherwise.
        """

        if self.pjs._copied_poly_ is None or self.pjs._copied_poly_ == []:
            return False

        if paste_shifted:
            copied_poly = (RUtils.qpolygonf2ndarray(self.pjs._copied_poly_) + RCBAnnotations.PIX_SHIFT).tolist()
        else:
            copied_poly = RUtils.qpolygonf2list(self.pjs._copied_poly_)

        self.pjs.addPolyline(copied_poly, self.pjs.curslice, pushundo=True)

        return True

    def cbMovePolyline(self) -> bool:
        """
        Set annotation mode to move the polyline clicked on.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.move_polyline
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbDilateErodePolyline(self) -> bool:
        """
        Set annotation mode to dilate or erode the polyline clicked on.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.select_polyline_erosion_dilation
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbTrackFiducials(self, firstSlice: Optional[int] = None, lastSlice: Optional[int] = None, wait_for_thread: bool = False) -> bool:
        """
        Match fiducials across slices in a stack based on minimum distance. Requires a constant number of fiducials. Runs in a different thread.

        :param firstSlice: start slice (>=1).
        :param lastSlice: final slice.
        :param wait_for_thread: True if PyJAMAS should wait for the function to finish before returning control to the user, False otherwise.
        :return: True if tracking finished correctly, False otherwise.
        """

        # If not enough parameters, open a dialog.
        if (firstSlice is False or firstSlice is None or lastSlice is False or lastSlice is None) \
                and self.pjs is not None:
            dialog = QtWidgets.QDialog()
            ui = dialogs.timepoints.TimePointsDialog()

            firstSlice = self.pjs.curslice + 1
            lastSlice = 1 if self.pjs.n_frames == 1 else self.pjs.slices.shape[0]

            ui.setupUi(dialog, firstslice=firstSlice, lastslice=lastSlice)
            dialog.exec()
            dialog.show()
            # If the dialog was closed by pressing OK, then run the measurements.
            continue_flag = dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
            theparameters_tuple = ui.parameters()
            theparameters = {'first': theparameters_tuple[0], 'last': theparameters_tuple[1]}

            dialog.close()

        else:
            theparameters = {'first': firstSlice, 'last': lastSlice}
            continue_flag = True

        # When you have the input parameters:
        if continue_flag:
            # Track forward.
            if theparameters['first'] <= theparameters['last']:
                theslicenumbers = numpy.arange(theparameters['first'] - 1, theparameters['last'])

            # Track backwards.
            else:
                theslicenumbers = numpy.arange(theparameters['first'] - 1, theparameters['last'] - 2, -1)

            # But DO track (in a thread)!!
            self.launch_thread(self.track_fiducials, {'theslices': theslicenumbers, 'progress': True, 'stop': True},
                               finished_fn=self.finished_fn, stop_fn=self.stop_fn, progress_fn=self.progress_fn,
                               wait_for_thread=wait_for_thread)

            return True
        else:
            return False

    def track_fiducials(self, theslices: numpy.ndarray, progress_signal: ThreadSignals, stop_signal: ThreadSignals) -> bool:

        #Requires a constant number of fiducials.

        #:param theslices: indeces of the slices in which fiducials will be tracked. They do not need to be consecutive.
        #:return:

        # Make sure that the slices are in a 1D numpy array.
        theslices = numpy.atleast_1d(theslices)
        num_slices = theslices.size

        # For every slice ...
        for i in range(num_slices-1):
            if len(self.pjs.fiducials[theslices[i]]) == 0:
                if stop_signal is not None:
                    stop_signal.emit(f"Stopping at slice {theslices[i]+1}: there are no fiducials to track there!")
                return True

            if len(self.pjs.fiducials[theslices[i+1]]) == 0:
                if stop_signal is not None:
                    stop_signal.emit(f"Stopping at slice {theslices[i+1]+1}: there are no fiducials to track there!")
                return True

            if len(self.pjs.fiducials[theslices[i]]) != len(self.pjs.fiducials[theslices[i+1]]):
                if stop_signal is not None:
                    stop_signal.emit(f"Error: slices {theslices[i]+1} and {theslices[i + 1] + 1} have a different number of fiducials.")
                return False

            fiducials_orig = numpy.array(self.pjs.fiducials[theslices[i]])
            fiducials_dest = numpy.array(self.pjs.fiducials[theslices[i + 1]])

            # Find distance between all pairs of fiducials.
            distance_matrix: numpy.ndarray = RUtils.point2point_distances(fiducials_orig, fiducials_dest)
            sorted_indices: numpy.ndarray = distance_matrix.argsort()

            # For each fiducials_orig[ii] select the closest fiducials_dest[jj].
            closest_fiducial_index: numpy.ndarray = sorted_indices[:, 0]

            # Deal with cases in which more than one fiducial maps to another.
            # bincount is WAY faster than unique.
            #theindex, counts = numpy.unique(closest_fiducial_index, return_counts=True)
            #conflictive_indeces = theindex[counts > 1]
            counts: int = numpy.bincount(closest_fiducial_index)

            if any(counts > 1):
                conflictive_indeces = numpy.arange(counts.shape[0])[counts > 1]
                original_fiducials = numpy.arange(closest_fiducial_index.shape[0])[closest_fiducial_index == conflictive_indeces[0]]

                if stop_signal is not None:
                    stop_signal.emit(f"Error in slice {theslices[i + 1]+1}, fiducial {conflictive_indeces[0]+1}: fiducials {original_fiducials+1} from slice {theslices[i]+1} map here.")
                return False

            else:
                self.pjs.fiducials[theslices[i+1]] = fiducials_dest[closest_fiducial_index].tolist()

            if progress_signal is not None:
                progress_signal.emit(int((100*(i+1))/(num_slices-1)))

        return True

    def cbDeleteAllAnn(self) -> bool:
        """
        Delete annotations from all image slices.

        :return: True.
        """
        self.pjs.undo_stack.push({'changetype': undo_modes.MULTI, 'frame': (0, self.pjs.n_frames), 'index': (True, True, True), 'details': self.pjs.copy_annotations()})
        self.pjs.fiducials = [[] for _ in range(self.pjs.n_frames)]
        self.pjs.polylines = [[] for _ in range(self.pjs.n_frames)]
        self.pjs.polyline_ids = [[] for _ in range(self.pjs.n_frames)]

        self.pjs.repaint()

        return True

    def cbDeleteSliceAnn(self, index: Optional[int] = None) -> bool:
        """
        Delete annotations from a specific slice.

        :param index: index of the slice in which annotations will be deleted (>=0). Defaults to the current slice (pjs.curslice).
        :return: True.
        """

        if index is None or index is False:
            index = self.pjs.curslice

        self.pjs.undo_stack.push({'changetype': undo_modes.MULTI, 'frame': (index, index + 1), 'index': (True, True, True), 'details': self.pjs.copy_annotations((index, index+1))})
        self.pjs.fiducials[index] = []
        self.pjs.polylines[index] = []
        self.pjs.polyline_ids[index] = []

        self.pjs.repaint()

        return True

    def cbDeleteSlicePoly(self, index: Optional[int] = None) -> bool:
        """
        Delete polylines from a specific slice.

        :param index: index of the slice in which polylines will be deleted (>=0). Defaults to the current slice (pjs.curslice).
        :return: True.
        """

        if index is None or index is False:
            index = self.pjs.curslice

        self.pjs.undo_stack.push({'changetype': undo_modes.MULTI, 'frame': (index, index + 1), 'index': (False, True, True), 'details': self.pjs.copy_annotations((index, index+1))})
        self.pjs.polylines[index] = []
        self.pjs.polyline_ids[index] = []

        self.pjs.repaint()

        return True

    def cbDeleteSliceFiducials(self, index: Optional[int] = None) -> bool:
        """
        Delete fiducials from a specific slice.

        :param index: index of the slice in which fiducials will be deleted (>=0). Defaults to the current slice (pjs.curslice).
        :return: True
        """

        if index is None or index is False:
            index = self.pjs.curslice

        self.pjs.undo_stack.push({'changetype': undo_modes.MULTI, 'frame': (index, index + 1), 'index': (True, False, False), 'details': self.pjs.copy_annotations((index, index+1))})
        self.pjs.fiducials[index] = []

        self.pjs.repaint()

        return True

    def cbDeleteAllFiducials(self) -> bool:
        """
        Delete all fiducials from all slices.

        :return: True
        """

        self.pjs.undo_stack.push({'changetype': undo_modes.MULTI, 'frame': (0, self.pjs.n_frames), 'index': (True, False, False), 'details': self.pjs.copy_annotations()})
        self.pjs.fiducials = [[] for _ in range(self.pjs.n_frames)]

        self.pjs.repaint()

        return True

    def cbDeleteFiducialsOutsidePoly(self) -> bool:
        """
        Set annotation mode to remove fiducials outside the polyline clicked on.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.delete_fiducials_outside_polyline
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

        return True

    def cbDeleteFiducialsInsidePoly(self) -> bool:
        """
        Set annotation mode to remove fiducials inside the polyline clicked on.

        :return: True.
        """

        self.pjs.annotation_mode = PyJAMAS.delete_fiducials_inside_polyline
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

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

    def process_polyline_roi(self, event, fn):
        """TODO: replace with appropriate functions in cbCrop, cbKymograph, and cbExportROIAndMasks"""
        """fn is one of self.pjs.image.cbCrop, self.pjs.image.cbKymograph and self.pjs.io.cbExportROIAndMasks"""
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            item_index = self.pjs.find_clicked_polyline(self.x, self.y)
            # ItemIndex of -1 means no polyline found, in which case, exit
            if item_index == -1:
                return False

            if fn is self.pjs.image.cbCrop and self.pjs.crop_tracked_polyline:  # Crop tracked polyline, input index to cbCrop
                thepolyline = numpy.array([item_index])
            else:  # Crop polyline on this slice, input min and max coordinates to cbCrop
                thepolyline = RUtils.qpolygonf2ndarray(self.pjs.polylines[self.pjs.curslice][item_index])
            fn(polyline=thepolyline, margin_size=self.pjs.margin_size)
        return True
