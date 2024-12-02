import os.path
from typing import List, Tuple

import pyjamas
import cv2
import numpy
import pytest
import skimage

FIXTURE_DIR: str = os.path.join(os.path.split(pyjamas.__file__)[0], "tests", "unit", "fixtures")
LOGO_FIXTURE: str = 'original_logo.tif'
IMAGE_FIXTURE: str = 'image.tif'
PUNCTA_FIXTURE: str = 'puncta.tif'
CLASSIFIER_FIXTURE: Tuple[str, str] = ('classifier_lr.cfr', 'classifier_svm.cfr')
DISPLAY_FIXTURE: str = 'display.tif'
BINMASKS_FIXTURE: str = 'binmasks.tif'
MOVIE_FIXTURE: str = 'movie.avi'
PJSANNOTATIONS_FIXTURE: str = 'annotations.pjs'
LOGOANNOTATIONS_FIXTURE: str = 'logoannotations.pjs'
REGISTRATIONANNOTATIONS_FIXTURE: str = 'logoregannotations.pjs'
MATANNOTATIONS_FIXTURE: str = 'annotations.mat'
SINGLE_CELL_PJSANNOTATIONS_FIXTURE: str = 'cell_00001.pjs'
PROJECTION_FIXTURE: Tuple[str, str] = ('max_projection.tif', 'sum_projection.tif')
ZPROJECTIONS_FIXTURE: Tuple[str, str] = ('xz.tif', 'yz.tif')
ROTATED_CW_FIXTURE: str = 'rotated_cw.tif'
FLIPPED_LR_FIXTURE: str = 'flipped_lr.tif'
FLIPPED_UD_FIXTURE: str = 'flipped_ud.tif'
INVERTED_FIXTURE: str = 'inverted.tif'
GRADIENT_FIXTURE: str = 'gradient.tif'
KYMOGRAPH_FIXTURE: str = 'kymograph.tif'
REGISTERED_FIXTURE: str = 'registered.tif'
TMP_IMG_FILE: str = 'image.tif'
TMP_DIR: str = os.path.join(os.path.split(pyjamas.__file__)[0], "tests", "tmp")
TMP_DISP_FILE: str = 'display.tif'
TMP_BINMASKS_FILE: str = 'binmasks.tif'
TMP_MOVIE_FILE: str = 'movie.avi'
TMP_PJS_FILE: str = 'annotations.pjs'
TMP_MAT_FILE: str = 'annotations.mat'
TMP_SINGLE_CELL_PJS_FILE: str = 'cell_00001.pjs'
PIX_PERCENTILE_LIMITS: Tuple[int, int] = (10, 90)
GAUSSIAN_SIGMA: float = 3.0
RESCALE_TUPLE: Tuple[float, float] = (3.0, 0.5)
FINDSEEDS_SIGMA: float = 2.0
FINDSEEDS_WINDOWSIZE: int = 16
FINDSEEDS_MINDIST: float = 3.0
FINDSEEDS_BINCLOSINGS: int = 2
FINDSEEDS_ANNOTATIONSPATH: str = 'fiducials_sigma2_window16_closings2_mindist3.pjs'
PROPAGATESEEDS_ANNOTATIONSPATH: str = 'propagatedfiducials_sigma2_window16_closings2_mindist3.pjs'
PROPAGATESEEDS0_ANNOTATIONSPATH: str = 'propagatedfiducials0_sigma2_window16_closings2_mindist3.pjs'
EXPANDSEEDS_ANNOTATIONSPATH: str = 'expandedfiducials_sigma2_window16_closings2_mindist3.pjs'
EXPANDNPROPAGATESEEDS_ANNOTATIONSPATH: str = 'expandedpropagatedfiducials_sigma2_window16_closings2_mindist3.pjs'
PUNCTA_ANNOTATIONSPATH: str = 'puncta_meanfilt25_th50_merge20.pjs'
PROJECT_CONCAT_DIR: str = 'projcat'
TMP_PROJCAT_FILE: str = 'projcat.tif'
BATCH_MEASURE_DIR: str = 'batch_measure/results'
BATCH_FLATFIELD_DIR: str = 'batch_ffc/images_for_correction'
BATCH_FLATFIELD_OUTPUTDIR: str = 'batch_ffc/corrected_images'
BATCH_FLATFIELD_FLATFIELDIMAGEPATH: str = 'batch_ffc/ffc_image.tif'
BATCH_FLATFIELD_DARKIMAGEPATH: str = 'batch_ffc/dark_field_image.tif'
BATCH_FLATFIELD_SUBSTR: str = ''
BATCH_FLATFIELD_FILESUFFIX: str = '_ffc_corrected'
BATCH_FLATFIELD_CROPDIMS: Tuple[int, int] = (512, 512)
PUNCTA_MEAN_FILTER_WIDTH: int = 25
PUNCTA_THRESHOLD: int = 50
PUNCTA_MAX_SIZE_MERGE: int = 20

@pytest.fixture
def image_fixture():
    return skimage.io.imread(os.path.join(FIXTURE_DIR, IMAGE_FIXTURE))


@pytest.fixture
def display_fixture():
    return skimage.io.imread(os.path.join(FIXTURE_DIR, DISPLAY_FIXTURE))


@pytest.fixture
def binmasks_fixture():
    return skimage.io.imread(os.path.join(FIXTURE_DIR, BINMASKS_FIXTURE))


@pytest.fixture
def pjsannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, PJSANNOTATIONS_FIXTURE)


@pytest.fixture
def logoannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, LOGOANNOTATIONS_FIXTURE)


@pytest.fixture
def singlecellpjsannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, SINGLE_CELL_PJSANNOTATIONS_FIXTURE)


@pytest.fixture
def matannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, MATANNOTATIONS_FIXTURE)


@pytest.fixture
def registrationannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, REGISTRATIONANNOTATIONS_FIXTURE)


@pytest.fixture
def imagepath_fixture():
    return os.path.join(FIXTURE_DIR, IMAGE_FIXTURE)


@pytest.fixture
def punctapath_fixture():
    return os.path.join(FIXTURE_DIR, PUNCTA_FIXTURE)


@pytest.fixture
def classifierpath_fixture():
    return (os.path.join(FIXTURE_DIR, classifier_fixture_filename) for classifier_fixture_filename in
            CLASSIFIER_FIXTURE)


@pytest.fixture
def movie_fixture():
    movie = cv2.VideoCapture(os.path.join(FIXTURE_DIR, MOVIE_FIXTURE))

    success, movie_array = movie.read()
    success, theslice = movie.read()

    while success:
        movie_array = numpy.concatenate((movie_array, theslice), axis=0)
        success, theslice = movie.read()

    return movie_array


@pytest.fixture
def rotatecwpath_fixture():
    return os.path.join(FIXTURE_DIR, ROTATED_CW_FIXTURE)


@pytest.fixture
def fliplrpath_fixture():
    return os.path.join(FIXTURE_DIR, FLIPPED_LR_FIXTURE)


@pytest.fixture
def flipudpath_fixture():
    return os.path.join(FIXTURE_DIR, FLIPPED_UD_FIXTURE)


@pytest.fixture
def invertpath_fixture():
    return os.path.join(FIXTURE_DIR, INVERTED_FIXTURE)


@pytest.fixture
def gradientpath_fixture():
    return os.path.join(FIXTURE_DIR, GRADIENT_FIXTURE)


@pytest.fixture
def projectionpath_fixture():
    return (os.path.join(FIXTURE_DIR, projection_fixture_filename) for projection_fixture_filename in
            PROJECTION_FIXTURE)


@pytest.fixture
def registeredpath_fixture():
    return os.path.join(FIXTURE_DIR, REGISTERED_FIXTURE)


@pytest.fixture
def contrastlimits_fixture():
    return PIX_PERCENTILE_LIMITS


@pytest.fixture
def gaussiansigma_fixture():
    return GAUSSIAN_SIGMA


@pytest.fixture
def kymographpath_fixture():
    return os.path.join(FIXTURE_DIR, KYMOGRAPH_FIXTURE)


@pytest.fixture
def rescaletuple_fixture():
    return RESCALE_TUPLE


@pytest.fixture
def zprojectionspath_fixture():
    return os.path.join(FIXTURE_DIR, ZPROJECTIONS_FIXTURE[0]), os.path.join(FIXTURE_DIR, ZPROJECTIONS_FIXTURE[1])


@pytest.fixture
def findseeds_parameters():
    return 1, 1, FINDSEEDS_SIGMA, FINDSEEDS_WINDOWSIZE, FINDSEEDS_BINCLOSINGS, FINDSEEDS_MINDIST


@pytest.fixture
def findseedsannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, FINDSEEDS_ANNOTATIONSPATH)


@pytest.fixture
def propagateseeds_parameters():
    return 1, 20, FINDSEEDS_WINDOWSIZE


@pytest.fixture
def propagateseeds0annotationspath_fixture():
    return os.path.join(FIXTURE_DIR, PROPAGATESEEDS0_ANNOTATIONSPATH)


@pytest.fixture
def propagateseedsannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, PROPAGATESEEDS_ANNOTATIONSPATH)


@pytest.fixture
def expandseedsannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, EXPANDSEEDS_ANNOTATIONSPATH)


@pytest.fixture
def expandseeds_parameters():
    return 1, 1, FINDSEEDS_SIGMA

@pytest.fixture
def expandnpropagateseedsannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, EXPANDNPROPAGATESEEDS_ANNOTATIONSPATH)


@pytest.fixture
def expandnpropagateseeds_parameters():
    return 1, 20, FINDSEEDS_SIGMA, FINDSEEDS_WINDOWSIZE


@pytest.fixture
def punctaannotationspath_fixture():
    return os.path.join(FIXTURE_DIR, PUNCTA_ANNOTATIONSPATH)


@pytest.fixture
def puncta_parameters():
    return PUNCTA_MEAN_FILTER_WIDTH, PUNCTA_THRESHOLD, PUNCTA_MAX_SIZE_MERGE


@pytest.fixture
def projcatfolder_fixture():
    return os.path.join(FIXTURE_DIR, PROJECT_CONCAT_DIR)


@pytest.fixture
def batchmeasurefolder_fixture():
    return os.path.join(FIXTURE_DIR, BATCH_MEASURE_DIR)

@pytest.fixture
def fieldcorrection_fixture():
    return {'input_folder': os.path.join(FIXTURE_DIR, BATCH_FLATFIELD_DIR),
            'darkfield_file': os.path.join(FIXTURE_DIR, BATCH_FLATFIELD_DARKIMAGEPATH),
            'flatfield_file': os.path.join(FIXTURE_DIR, BATCH_FLATFIELD_FLATFIELDIMAGEPATH),
            'crop_dims': BATCH_FLATFIELD_CROPDIMS,
            'file_suffix': BATCH_FLATFIELD_FILESUFFIX,
            'input_substr': BATCH_FLATFIELD_SUBSTR,
            'output_folder': os.path.join(FIXTURE_DIR, BATCH_FLATFIELD_OUTPUTDIR)
            }
