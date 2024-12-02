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

# tests for io menu
# for coverage, run:
# coverage run -m pytest
# or if you want to include branches:
# coverage run --branch -m pytest
# followed by:
# coverage report -i

import gzip
import os.path
import pickle

import cv2
import numpy
import pytest
import skimage
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from pyjamas.pjscore import PyJAMAS
from pyjamas.rutils import RUtils
import pyjamas.tests.unit.pjsfixtures as pjsfixtures

PyJAMAS_FIXTURE: PyJAMAS = PyJAMAS()
PyJAMAS_FIXTURE.options.cbSetCWD(pjsfixtures.TMP_DIR)


@pytest.mark.usefixtures("image_fixture")
def test_cbLoadArray(image_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, image_fixture)


@pytest.mark.usefixtures("imagepath_fixture")
@pytest.mark.usefixtures("image_fixture")
def test_cbLoadTimeSeries(imagepath_fixture, image_fixture):
    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(imagepath_fixture)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, image_fixture)


@pytest.mark.usefixtures("classifierpath_fixture")
def test_cbLoadClassifier(classifierpath_fixture):
    thepjscoefs: numpy.ndarray = None
    thetestcoefs: numpy.ndarray = None

    for aclassifier in classifierpath_fixture:
        PyJAMAS_FIXTURE.io.cbLoadClassifier(aclassifier)

        with gzip.open(aclassifier, "rb") as fh:
            theparameters = pickle.load(fh)
            theclassifier = theparameters.get('classifier')

            if type(theclassifier) is SVC:
                thepjscoefs = PyJAMAS_FIXTURE.batch_classifier.image_classifier.classifier.dual_coef_
                thetestcoefs = theclassifier.dual_coef_
            elif type(theclassifier) is LogisticRegression:
                thepjscoefs = PyJAMAS_FIXTURE.batch_classifier.image_classifier.classifier.coef_
                thetestcoefs = theclassifier.coef_

            assert numpy.array_equal(thepjscoefs, thetestcoefs)


@pytest.mark.usefixtures("classifierpath_fixture")
def test_cbSaveClassifier(classifierpath_fixture):
    thepjscoefs: numpy.ndarray = None
    thetestcoefs: numpy.ndarray = None

    for aclassifier, afilename in zip(classifierpath_fixture, pjsfixtures.CLASSIFIER_FIXTURE):
        PyJAMAS_FIXTURE.io.cbLoadClassifier(aclassifier)
        theclassifier = PyJAMAS_FIXTURE.batch_classifier.image_classifier.classifier
        if type(theclassifier) is SVC:
            thepjscoefs = theclassifier.dual_coef_
        elif type(theclassifier) is LogisticRegression:
            thepjscoefs = theclassifier.coef_

        PyJAMAS_FIXTURE.io.cbSaveClassifier(os.path.join(pjsfixtures.TMP_DIR, afilename))
        PyJAMAS_FIXTURE.io.cbLoadClassifier(os.path.join(pjsfixtures.TMP_DIR, afilename))

        theclassifier = PyJAMAS_FIXTURE.batch_classifier.image_classifier.classifier
        if type(theclassifier) is SVC:
            thetestcoefs = theclassifier.dual_coef_
        elif type(theclassifier) is LogisticRegression:
            thetestcoefs = theclassifier.coef_

        assert numpy.array_equal(thepjscoefs, thetestcoefs)


@pytest.mark.usefixtures("image_fixture")
def test_cbSaveTimeSeries(image_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbSaveTimeSeries(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_IMG_FILE))
    assert numpy.array_equal(image_fixture,
                             skimage.io.imread(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_IMG_FILE)))


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbSaveROI(image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])

    roi_coords = RUtils.qpolygonf2ndarray(PyJAMAS_FIXTURE.polylines[0][0])
    x_range = (int(min(roi_coords[:, 0])), int(max(roi_coords[:, 0])))
    y_range = (int(min(roi_coords[:, 1])), int(max(roi_coords[:, 1])))
    z_range = (0, 0)

    PyJAMAS_FIXTURE.io.cbSaveROI(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_IMG_FILE), x_range, y_range, z_range)

    assert numpy.array_equal(
        image_fixture[z_range[0]:z_range[1] + 1, y_range[0]:y_range[1] + 1, x_range[0]:x_range[1] + 1],
        skimage.io.imread(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_IMG_FILE)))


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbLoadAnnotations(image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])

    with gzip.open(pjsannotationspath_fixture, "rb") as fh:
        fiducials = pickle.load(fh)
        polylines = pickle.load(fh)

    assert numpy.array_equal(fiducials, PyJAMAS_FIXTURE.fiducials) and \
           numpy.array_equal(
               numpy.asarray([RUtils.qpolygonf2list(apolyline) for apolyline in polylines]),
               numpy.asarray([RUtils.qpolygonf2list(apolyline) for apolyline in PyJAMAS_FIXTURE.polylines]))


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbSaveAnnotations(image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])
    PyJAMAS_FIXTURE.io.cbSaveAnnotations(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_PJS_FILE))

    with gzip.open(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_PJS_FILE), "rb") as fh:
        fiducials = pickle.load(fh)
        polylines = pickle.load(fh)

    assert numpy.array_equal(fiducials, PyJAMAS_FIXTURE.fiducials) and \
           numpy.array_equal(
               numpy.asarray([RUtils.qpolygonf2list(apolyline) for apolyline in polylines]),
               numpy.asarray([RUtils.qpolygonf2list(apolyline) for apolyline in PyJAMAS_FIXTURE.polylines]))


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbExportPolylineAnnotations(image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])
    PyJAMAS_FIXTURE.image.cbExpandSeeds(1, 1, sigma=2.0,
                                        wait_for_thread=True)  # you need to wait, otherwise the test fails!!!
    PyJAMAS_FIXTURE.removePolylineByIndex(0, 0)

    PyJAMAS_FIXTURE.options.cbSetCWD(pjsfixtures.TMP_DIR)

    # mimic click on first fiducial.
    x, y = PyJAMAS_FIXTURE.fiducials[0][0]
    PyJAMAS_FIXTURE.io.export_polyline_annotations(x, y)

    with gzip.open(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_SINGLE_CELL_PJS_FILE), "rb") as fexp, \
            gzip.open(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_SINGLE_CELL_PJS_FILE), "rb") as ffix:
        fiducials_exp = pickle.load(fexp)
        polylines_exp = pickle.load(fexp)

        fiducials_fix = pickle.load(ffix)
        polylines_fix = pickle.load(ffix)

        assert fiducials_exp == fiducials_fix and \
               polylines_exp == polylines_fix


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
@pytest.mark.usefixtures("singlecellpjsannotationspath_fixture")
def test_cbExportAllPolylineAnnotations(image_fixture, pjsannotationspath_fixture,
                                        singlecellpjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])
    PyJAMAS_FIXTURE.image.cbExpandSeeds(1, 1, sigma=2.0,
                                        wait_for_thread=True)  # you need to wait, otherwise the test fails!!!
    PyJAMAS_FIXTURE.removePolylineByIndex(0, 0)
    PyJAMAS_FIXTURE.io.cbExportAllPolylineAnnotations(pjsfixtures.TMP_DIR)

    PyJAMAS_FIXTURE.io.cbLoadAnnotations([os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_SINGLE_CELL_PJS_FILE)])

    with gzip.open(singlecellpjsannotationspath_fixture, "rb") as fh:
        fiducials = pickle.load(fh)
        polylines = pickle.load(fh)

    assert fiducials == PyJAMAS_FIXTURE.fiducials and \
           polylines == RUtils.qpolygonfs2coordinatelists(PyJAMAS_FIXTURE.polylines)


@pytest.mark.usefixtures("matannotationspath_fixture")
@pytest.mark.usefixtures("imagepath_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbImportSIESTAAnnotations(matannotationspath_fixture, imagepath_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(imagepath_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])

    fiducials = PyJAMAS_FIXTURE.fiducials.copy()
    polylines = PyJAMAS_FIXTURE.polylines.copy()

    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(imagepath_fixture)
    PyJAMAS_FIXTURE.io.cbImportSIESTAAnnotations([matannotationspath_fixture])

    assert fiducials == PyJAMAS_FIXTURE.fiducials and \
           polylines == PyJAMAS_FIXTURE.polylines


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbExportSIESTAAnnotations(image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])

    fiducials = PyJAMAS_FIXTURE.fiducials.copy()
    polylines = PyJAMAS_FIXTURE.polylines.copy()

    PyJAMAS_FIXTURE.io.cbExportSIESTAAnnotations(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_PJS_FILE))
    PyJAMAS_FIXTURE.io.cbImportSIESTAAnnotations([os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_PJS_FILE)])

    assert fiducials == PyJAMAS_FIXTURE.fiducials and \
           polylines == PyJAMAS_FIXTURE.polylines

@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbLoadAnnotations_additive(image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])

    assert PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture], replace=False)


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("matannotationspath_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbImportSIESTAAnnotations_additive(image_fixture, matannotationspath_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture], replace=False)

    fiducials = PyJAMAS_FIXTURE.fiducials.copy()
    polylines = PyJAMAS_FIXTURE.polylines.copy()

    PyJAMAS_FIXTURE.io.cbImportSIESTAAnnotations([matannotationspath_fixture])
    PyJAMAS_FIXTURE.io.cbImportSIESTAAnnotations([matannotationspath_fixture], replace=False)

    assert numpy.array_equal(fiducials, PyJAMAS_FIXTURE.fiducials) and \
           numpy.array_equal(
               numpy.asarray([RUtils.qpolygonf2list(apolyline) for apolyline in polylines]),
               numpy.asarray([RUtils.qpolygonf2list(apolyline) for apolyline in PyJAMAS_FIXTURE.polylines]))


@pytest.mark.usefixtures("display_fixture")
@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbSaveDisplay(display_fixture, image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])
    PyJAMAS_FIXTURE.io.cbSaveDisplay(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_DISP_FILE))
    assert numpy.array_equal(display_fixture,
                             skimage.io.imread(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_DISP_FILE)))


@pytest.mark.usefixtures("movie_fixture")
@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbExportMovie(movie_fixture, image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])
    PyJAMAS_FIXTURE.io.cbExportMovie(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_MOVIE_FILE))

    movie = cv2.VideoCapture(os.path.join(pjsfixtures.FIXTURE_DIR, pjsfixtures.MOVIE_FIXTURE))

    success, movie_array = movie.read()
    success, slice = movie.read()

    while success:
        movie_array = numpy.concatenate((movie_array, slice), axis=0)
        success, slice = movie.read()

    assert numpy.array_equal(movie_fixture, movie_array)


@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
@pytest.mark.usefixtures("binmasks_fixture")
def test_cbExportROIAndMasks(image_fixture, pjsannotationspath_fixture, binmasks_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])
    PyJAMAS_FIXTURE.image.cbExpandSeeds(1, 1, sigma=2.0,
                                        wait_for_thread=True)  # you need to wait, otherwise the test fails!!!
    PyJAMAS_FIXTURE.removePolylineByIndex(0, 0)

    PyJAMAS_FIXTURE.options.cbSetCWD(pjsfixtures.TMP_DIR)

    PyJAMAS_FIXTURE.io.cbExportROIAndMasks(
        numpy.array([[0, 0], [PyJAMAS_FIXTURE.width - 1, PyJAMAS_FIXTURE.height - 1]]))

    filenameext = os.path.join(
        PyJAMAS_FIXTURE.io.generate_ROI_filename((0, PyJAMAS_FIXTURE.width - 1), (0, PyJAMAS_FIXTURE.height - 1),
                                                 (PyJAMAS_FIXTURE.curslice, PyJAMAS_FIXTURE.curslice),
                                                 PyJAMAS.image_extensions[0], relative=True))
    filename, _ = os.path.splitext(filenameext)
    path_to_image: str = os.path.join(pjsfixtures.TMP_DIR, filename, "image", filenameext)
    path_to_mask: str = os.path.join(pjsfixtures.TMP_DIR, filename, "mask", filenameext)

    assert numpy.array_equal(image_fixture[PyJAMAS_FIXTURE.curslice:PyJAMAS_FIXTURE.curslice + 1],
                             skimage.io.imread(path_to_image)) and \
           numpy.array_equal(binmasks_fixture, skimage.io.imread(path_to_mask))


@pytest.mark.usefixtures("binmasks_fixture")
@pytest.mark.usefixtures("image_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
def test_cbExportCurrentAnnotationsBinaryImage(binmasks_fixture, image_fixture, pjsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadArray(image_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([pjsannotationspath_fixture])

    PyJAMAS_FIXTURE.image.cbExpandSeeds(1, 1, sigma=2.0,
                                        wait_for_thread=True)  # you need to wait, otherwise the test fails!!!
    PyJAMAS_FIXTURE.removePolylineByIndex(0, 0)

    PyJAMAS_FIXTURE.io.cbExportCurrentAnnotationsBinaryImage(
        os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_BINMASKS_FILE),
        numpy.array([[0, 0], [PyJAMAS_FIXTURE.width - 1,
                              PyJAMAS_FIXTURE.height - 1]]),
        firstSlice=PyJAMAS_FIXTURE.curslice + 1, lastSlice=PyJAMAS_FIXTURE.curslice + 1)

    assert numpy.array_equal(binmasks_fixture,
                             skimage.io.imread(os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_BINMASKS_FILE)))


PyJAMAS_FIXTURE.app.quit()
