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

# tests for image menu
# for coverage, run:
# coverage run -m pytest
# or if you want to include branches:
# coverage run --branch -m pytest
# followed by:
# coverage report -i

import gzip
import pickle

import numpy
import os
import pytest
import skimage

from pyjamas.pjscore import PyJAMAS
from pyjamas.rimage.rimutils import rimutils
from pyjamas.rutils import RUtils
from pyjamas.rcallbacks.rcbimage import projection_types
import pyjamas.tests.unit.pjsfixtures as pjsfixtures

PyJAMAS_FIXTURE: PyJAMAS = PyJAMAS()
PyJAMAS_FIXTURE.io.cbLoadTimeSeries(os.path.join(pjsfixtures.FIXTURE_DIR, pjsfixtures.LOGO_FIXTURE))
PyJAMAS_FIXTURE.options.cbSetCWD(pjsfixtures.TMP_DIR)


@pytest.mark.usefixtures("contrastlimits_fixture")
def test_cbAdjustContrast(contrastlimits_fixture):
    PyJAMAS_FIXTURE.image.cbAdjustContrast(contrastlimits_fixture[0], contrastlimits_fixture[1])

    theqimage = PyJAMAS_FIXTURE._imageItem.pixmap().toImage()
    displayed_image = numpy.empty((PyJAMAS_FIXTURE.height, PyJAMAS_FIXTURE.width), dtype=numpy.uint8)

    for row_index in range(PyJAMAS_FIXTURE.height):
        for col_index in range(PyJAMAS_FIXTURE.width):
            displayed_image[row_index, col_index] = theqimage.pixel(col_index, row_index)

    assert numpy.array_equal(displayed_image,
                             numpy.array(
                                 rimutils.stretch(PyJAMAS_FIXTURE.imagedata,
                                                  contrastlimits_fixture[0],
                                                  contrastlimits_fixture[1]),
                                 dtype=numpy.uint8))


def test_cbPlay():
    assert PyJAMAS_FIXTURE.image.cbPlay()


@pytest.mark.usefixtures("rotatecwpath_fixture")
def test_cbRotateImage(rotatecwpath_fixture):
    original_image = PyJAMAS_FIXTURE.slices.copy()

    PyJAMAS_FIXTURE.image.cbRotateImage(PyJAMAS_FIXTURE.image.CW)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, skimage.io.imread(rotatecwpath_fixture))

    PyJAMAS_FIXTURE.image.cbRotateImage(PyJAMAS_FIXTURE.image.CCW)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, original_image)


@pytest.mark.usefixtures("fliplrpath_fixture", "flipudpath_fixture")
def test_cbFlipImage(fliplrpath_fixture, flipudpath_fixture):
    original_image = PyJAMAS_FIXTURE.slices.copy()

    PyJAMAS_FIXTURE.image.cbFlipImage(PyJAMAS_FIXTURE.image.LEFT_RIGHT)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, skimage.io.imread(fliplrpath_fixture))

    PyJAMAS_FIXTURE.image.cbFlipImage(PyJAMAS_FIXTURE.image.LEFT_RIGHT)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, original_image)

    PyJAMAS_FIXTURE.image.cbFlipImage(PyJAMAS_FIXTURE.image.UP_DOWN)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, skimage.io.imread(flipudpath_fixture))

    PyJAMAS_FIXTURE.image.cbFlipImage(PyJAMAS_FIXTURE.image.UP_DOWN)
    assert numpy.array_equal(PyJAMAS_FIXTURE.slices, original_image)


@pytest.mark.usefixtures("invertpath_fixture")
def test_cbInvertImage(invertpath_fixture):
    PyJAMAS_FIXTURE.image.cbInvertImage()
    theimage = PyJAMAS_FIXTURE.slices.copy()
    PyJAMAS_FIXTURE.options.cbUndo()

    assert numpy.array_equal(theimage, skimage.io.imread(invertpath_fixture))


@pytest.mark.usefixtures("gaussiansigma_fixture")
def test_cbGaussianImage(gaussiansigma_fixture):
    PyJAMAS_FIXTURE.image.cbGaussianImage(gaussiansigma_fixture)
    theimage = PyJAMAS_FIXTURE.slices.copy()
    PyJAMAS_FIXTURE.options.cbUndo()

    assert numpy.array_equal(theimage, rimutils.gaussian_smoothing(PyJAMAS_FIXTURE.slices, gaussiansigma_fixture))


@pytest.mark.usefixtures("gradientpath_fixture")
def test_cbGradientImage(gradientpath_fixture):
    PyJAMAS_FIXTURE.image.cbGradientImage()
    theimage = PyJAMAS_FIXTURE.slices.copy()
    PyJAMAS_FIXTURE.options.cbUndo()

    assert numpy.array_equal(theimage, skimage.io.imread(gradientpath_fixture))


@pytest.mark.usefixtures("projectionpath_fixture")
def test_cbProjectImage(projectionpath_fixture):
    for aprojectionpath, aprojectiontype in zip(projectionpath_fixture, projection_types):
        PyJAMAS_FIXTURE.image.cbProjectImage([], aprojectiontype)
        theproj = PyJAMAS_FIXTURE.slices.copy()
        PyJAMAS_FIXTURE.options.cbUndo()

        assert numpy.array_equal(theproj, skimage.io.imread(aprojectionpath))


@pytest.mark.usefixtures("rescaletuple_fixture")
def test_cbRescaleImage(rescaletuple_fixture):
    PyJAMAS_FIXTURE.image.cbRescaleImage(rescaletuple_fixture)
    theimage = PyJAMAS_FIXTURE.slices.copy()
    PyJAMAS_FIXTURE.options.cbUndo()

    assert numpy.array_equal(theimage, rimutils.rescale(PyJAMAS_FIXTURE.slices, rescaletuple_fixture))


@pytest.mark.usefixtures("registrationannotationspath_fixture")
@pytest.mark.usefixtures("registeredpath_fixture")
def test_cbRegisterImage(registrationannotationspath_fixture, registeredpath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([registrationannotationspath_fixture])
    PyJAMAS_FIXTURE.image.cbRegisterImage()
    theimage = PyJAMAS_FIXTURE.slices.copy()
    PyJAMAS_FIXTURE.options.cbUndo()

    assert numpy.array_equal(theimage, skimage.io.imread(registeredpath_fixture))


def test_cbCropImage():
    theimage = PyJAMAS_FIXTURE.slices.copy()

    # crop around the entire image.
    PyJAMAS_FIXTURE.image.cbCrop(numpy.asarray(
        [[0, 0], [PyJAMAS_FIXTURE.width, 0], [PyJAMAS_FIXTURE.width, PyJAMAS_FIXTURE.height],
         [0, PyJAMAS_FIXTURE.height], [0, 0]]))

    assert numpy.array_equal(theimage, PyJAMAS_FIXTURE.slices)


@pytest.mark.usefixtures("logoannotationspath_fixture")
@pytest.mark.usefixtures("kymographpath_fixture")
def test_cbKymograph(logoannotationspath_fixture, kymographpath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([logoannotationspath_fixture])

    PyJAMAS_FIXTURE.image.cbKymograph(RUtils.qpolygonf2ndarray(PyJAMAS_FIXTURE.polylines[0][0]))
    theimage = PyJAMAS_FIXTURE.slices.copy()

    PyJAMAS_FIXTURE.options.cbUndo()

    assert numpy.array_equal(theimage, skimage.io.imread(kymographpath_fixture))


@pytest.mark.usefixtures("zprojectionspath_fixture")
def test_cbOrthogonalViews(zprojectionspath_fixture):
    PyJAMAS_FIXTURE.image.cbOrthogonalViews()

    assert PyJAMAS_FIXTURE.orthogonal_views and PyJAMAS_FIXTURE.slicetracker

    zx, zy = PyJAMAS_FIXTURE.orthogonal_views.zx.copy(), PyJAMAS_FIXTURE.orthogonal_views.zy.copy()

    PyJAMAS_FIXTURE.image.cbOrthogonalViews()

    assert numpy.array_equal(zx, skimage.io.imread(zprojectionspath_fixture[0]))
    assert numpy.array_equal(zy, skimage.io.imread(zprojectionspath_fixture[1]))

    assert (not PyJAMAS_FIXTURE.orthogonal_views) and (not PyJAMAS_FIXTURE.slicetracker)


def test_cbZoom():
    for izoom in range(len(PyJAMAS_FIXTURE.zoom_factors)):
        PyJAMAS_FIXTURE.image.cbZoom(izoom)
        assert PyJAMAS_FIXTURE.zoom_index == izoom


def test_cbNextFrame():
    theimage = PyJAMAS_FIXTURE.slices[PyJAMAS_FIXTURE.curslice + 1]
    PyJAMAS_FIXTURE.image.cbNextFrame()

    assert numpy.array_equal(theimage, PyJAMAS_FIXTURE.imagedata)


# this should always be after test_cbNextFrame
def test_cbPrevFrame():
    theimage = PyJAMAS_FIXTURE.slices[PyJAMAS_FIXTURE.curslice - 1]
    PyJAMAS_FIXTURE.image.cbPrevFrame()

    assert numpy.array_equal(theimage, PyJAMAS_FIXTURE.imagedata)


def test_cbGoTo():
    theimage = PyJAMAS_FIXTURE.slices[-1]
    PyJAMAS_FIXTURE.image.cbGoTo(PyJAMAS_FIXTURE.n_frames - 1)
    assert numpy.array_equal(theimage, PyJAMAS_FIXTURE.imagedata)

    theimage = PyJAMAS_FIXTURE.slices[0]
    PyJAMAS_FIXTURE.image.cbGoTo(0)
    assert numpy.array_equal(theimage, PyJAMAS_FIXTURE.imagedata)


@pytest.mark.usefixtures("imagepath_fixture")
@pytest.mark.usefixtures("pjsannotationspath_fixture")
@pytest.mark.usefixtures("findseeds_parameters")
@pytest.mark.usefixtures("findseedsannotationspath_fixture")
def test_cbFindSeeds(imagepath_fixture, pjsannotationspath_fixture, findseeds_parameters,
                     findseedsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(imagepath_fixture)
    PyJAMAS_FIXTURE.image.cbFindSeeds(findseeds_parameters[0], findseeds_parameters[1], findseeds_parameters[2],
                                      findseeds_parameters[3], findseeds_parameters[4], findseeds_parameters[5],
                                      preview=False,
                                      wait_for_thread=True)

    with gzip.open(findseedsannotationspath_fixture, "rb") as fh:
        fiducials = pickle.load(fh)

    assert numpy.array_equal(fiducials[0], PyJAMAS_FIXTURE.fiducials[0])


@pytest.mark.usefixtures("imagepath_fixture")
@pytest.mark.usefixtures("propagateseeds0annotationspath_fixture")
@pytest.mark.usefixtures("propagateseeds_parameters")
@pytest.mark.usefixtures("propagateseedsannotationspath_fixture")
def test_cbPropagateSeeds(imagepath_fixture, propagateseeds0annotationspath_fixture, propagateseeds_parameters,
                          propagateseedsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(imagepath_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([propagateseeds0annotationspath_fixture])
    PyJAMAS_FIXTURE.image.cbPropagateSeeds(propagateseeds_parameters[0], propagateseeds_parameters[1],
                                           propagateseeds_parameters[2], wait_for_thread=True)

    with gzip.open(propagateseedsannotationspath_fixture, "rb") as fh:
        fiducials = pickle.load(fh)

    assert numpy.array_equal(fiducials, PyJAMAS_FIXTURE.fiducials)


@pytest.mark.usefixtures("imagepath_fixture")
@pytest.mark.usefixtures("propagateseeds0annotationspath_fixture")
@pytest.mark.usefixtures("expandseeds_parameters")
@pytest.mark.usefixtures("expandseedsannotationspath_fixture")
def test_cbExpandSeeds(imagepath_fixture, propagateseeds0annotationspath_fixture, expandseeds_parameters,
                       expandseedsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(imagepath_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([propagateseeds0annotationspath_fixture])
    PyJAMAS_FIXTURE.image.cbExpandSeeds(expandseeds_parameters[0], expandseeds_parameters[1], expandseeds_parameters[2],
                                        wait_for_thread=True)

    polylines = PyJAMAS_FIXTURE.polylines.copy()
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([expandseedsannotationspath_fixture])

    assert polylines == PyJAMAS_FIXTURE.polylines


@pytest.mark.usefixtures("imagepath_fixture")
@pytest.mark.usefixtures("propagateseeds0annotationspath_fixture")
@pytest.mark.usefixtures("expandnpropagateseeds_parameters")
@pytest.mark.usefixtures("expandnpropagateseedsannotationspath_fixture")
def test_cbExpandNPropagateSeeds(imagepath_fixture, propagateseeds0annotationspath_fixture, expandnpropagateseeds_parameters,
                       expandnpropagateseedsannotationspath_fixture):
    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(imagepath_fixture)
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([propagateseeds0annotationspath_fixture])
    PyJAMAS_FIXTURE.image.cbExpandNPropagateSeeds(expandnpropagateseeds_parameters[0],
                                                  expandnpropagateseeds_parameters[1],
                                                  expandnpropagateseeds_parameters[2],
                                                  expandnpropagateseeds_parameters[3],
                                                  wait_for_thread=True)

    polylines = PyJAMAS_FIXTURE.polylines.copy()
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([expandnpropagateseedsannotationspath_fixture])

    assert polylines == PyJAMAS_FIXTURE.polylines


@pytest.mark.usefixtures("punctapath_fixture")
@pytest.mark.usefixtures("punctaannotationspath_fixture")
@pytest.mark.usefixtures("puncta_parameters")
def test_cbFindPuncta(punctapath_fixture, punctaannotationspath_fixture, puncta_parameters):
    PyJAMAS_FIXTURE.io.cbLoadTimeSeries(punctapath_fixture)
    PyJAMAS_FIXTURE.image.cbFindPuncta(firstSlice=1, lastSlice=PyJAMAS_FIXTURE.n_frames,
                                       mean_filter_width=puncta_parameters[0],
                                       difference_threshold=puncta_parameters[1],
                                       max_size_merge=puncta_parameters[2],
                                       wait_for_thread=True)

    polylines = PyJAMAS_FIXTURE.polylines.copy()
    PyJAMAS_FIXTURE.io.cbLoadAnnotations([punctaannotationspath_fixture])

    assert polylines == PyJAMAS_FIXTURE.polylines


def test_cbDisplayInfo():
    assert PyJAMAS_FIXTURE.image.cbDisplayInfo()


PyJAMAS_FIXTURE.app.quit()
