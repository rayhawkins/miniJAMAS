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
# coverage run -m pytest -s
# or if you want to include branches:
# coverage run --branch -m pytest
# followed by:
# coverage report -i

# without these two instructions, pytest seg-faults with "Garbage-collecting" as the error message when running test_cbBatchMeasure
import gc
gc.disable()

from datetime import datetime
import os
from typing import List

import numpy
import pandas
import pytest
import skimage

from pyjamas.pjscore import PyJAMAS
from pyjamas.rimage.rimutils import rimutils
from pyjamas.rutils import RUtils
from pyjamas.rcallbacks.rcbimage import projection_types
import pyjamas.tests.unit.pjsfixtures as pjsfixtures

PyJAMAS_FIXTURE: PyJAMAS = PyJAMAS()
PyJAMAS_FIXTURE.options.cbSetCWD(pjsfixtures.TMP_DIR)


@pytest.mark.usefixtures("projcatfolder_fixture")
def test_cbBatchProjectConcat(projcatfolder_fixture):
    output_file: str = os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.TMP_PROJCAT_FILE)

    for aprojectiontype in projection_types:
        if aprojectiontype == projection_types.MAX:
            proj_fn = rimutils.mip
        elif aprojectiontype == projection_types.SUM:
            proj_fn = rimutils.sip

        PyJAMAS_FIXTURE.batch.cbBatchProjectConcat(projcatfolder_fixture, None, output_file, aprojectiontype, wait_for_thread=True)

        projected_array: numpy.ndarray = None

        for afile in RUtils.extract_file_paths(projcatfolder_fixture, ['.tif']):
            _, fname = os.path.split(afile)
            im3D = numpy.expand_dims(proj_fn(rimutils.read_stack(afile)), axis=0)

            if projected_array is not None:
                projected_array = numpy.concatenate((projected_array, im3D), axis=0)

            else:
                projected_array = im3D.copy()

        assert numpy.array_equal(rimutils.read_stack(output_file), projected_array)

@pytest.mark.usefixtures("fieldcorrection_fixture")
def test_cbBatchFlatFieldCorrection(fieldcorrection_fixture):
    files_to_correct = RUtils.extract_file_paths(fieldcorrection_fixture.get('input_folder'),
                                                 PyJAMAS_FIXTURE.batch.VALID_EXTENSIONS)
    files_corrected = RUtils.extract_file_paths(fieldcorrection_fixture.get('output_folder'),
                                                PyJAMAS_FIXTURE.batch.VALID_EXTENSIONS)

    PyJAMAS_FIXTURE.batch.cbBatchFlatFieldCorrection(fieldcorrection_fixture)

    for afile_pre, afile_post in zip(files_to_correct, files_corrected):
        im_post = rimutils.read_stack(afile_post)
        folder_name, full_file_name = os.path.split(afile_pre)
        file_name, ext = os.path.splitext(full_file_name)
        test_file_path = os.path.join(folder_name, file_name + fieldcorrection_fixture.get('file_suffix') + ext)
        im_test = rimutils.read_stack(test_file_path)
        os.remove(test_file_path)

        assert numpy.array_equal(im_post, im_test)
@pytest.mark.usefixtures("batchmeasurefolder_fixture")
def test_cbBatchMeasure(batchmeasurefolder_fixture):
    output_folder: str = os.path.join(pjsfixtures.TMP_DIR, pjsfixtures.BATCH_MEASURE_DIR)
    results_str_index: str = "'results_folder': "
    intensity_str_index: str = ", 'intensity_flag': "

    # Find the notebook files in the output folder, and generate equivalent ones to compare in the tmp folder.
    csv_files: List[str] = RUtils.extract_file_paths(batchmeasurefolder_fixture,[PyJAMAS_FIXTURE.batch.ANALYSIS_EXTENSION_BM])

    for afile in csv_files:
        file_prefix: str = afile[:-len(PyJAMAS_FIXTURE.batch.ANALYSIS_EXTENSION_BM)-len(PyJAMAS_FIXTURE.batch.ANALYSIS_FILENAME_APPENDIX_BM)]

        # There is an X_analysis_script.py file for each X_analysis.csv file. The .py file contains the code ran to execute the analysis.
        python_file: str = file_prefix + PyJAMAS_FIXTURE.batch.SCRIPT_FILENAME_APPENDIX_BM
        python_file = python_file[:-len(PyJAMAS.notebook_extension)] + PyJAMAS_FIXTURE.batch.SCRIPT_EXTENSION_BM

        if os.path.exists(python_file):
            with open(python_file, mode="r", encoding="utf-8") as thefile:
                code = thefile.read()

                # Change the output folder to be output_folder.
                ind_results_folder = code.find(results_str_index)+len(results_str_index)
                ind_intensity_flag = code.find(intensity_str_index)
                new_code: str = code[:ind_results_folder] + f"'{output_folder}'" + code[ind_intensity_flag:]

                exec(new_code)

                # open fixture notebook and the one just generated, run them and compare cell outputs, or ...
                # compare csv files!!!!
                fixture_df = pandas.read_csv(afile, index_col=0)

                test_csv_files: List[str] = RUtils.extract_file_paths(output_folder, [PyJAMAS_FIXTURE.batch.ANALYSIS_EXTENSION_BM])
                test_df = pandas.read_csv(test_csv_files[-1], index_col=0)

                assert test_df.equals(fixture_df)

PyJAMAS_FIXTURE.app.quit()
