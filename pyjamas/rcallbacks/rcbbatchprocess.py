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

from datetime import datetime
from enum import IntEnum
from itertools import chain
import os
from typing import List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import nbformat as nbf
from nbformat.notebooknode import NotebookNode
import numpy
import pandas as pd
pd.options.mode.copy_on_write = True  # avoid "setting with copy" warnings: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# import pylustrator
from PyQt6 import QtWidgets
import seaborn as sns
from scipy.optimize import leastsq
from scipy import stats
import skimage.io as sio
import skimage.transform as st

from pyjamas.dialogs.batchanalysis import BatchMeasureDialog
from pyjamas.dialogs.batchflatfield import BatchFlatFieldCorrectionDialog
from pyjamas.dialogs.batchresize import BatchResizeDialog
from pyjamas.dialogs.batchprojectconcat import BatchProjectConcatenateDialog
from pyjamas.pjscore import PyJAMAS
from pyjamas.pjsthreads import ThreadSignals
from pyjamas.rcallbacks.rcallback import RCallback
from pyjamas.rcallbacks.rcbimage import projection_types
from pyjamas.rimage.rimutils import rimutils as rimutils
from pyjamas.rutils import RUtils


class normalization_modes(IntEnum):
    #
    # 0 is raw intensities; 1 normalizes for photobleaching; 2 subtracts background -image mode-, normalizes for photobleaching, and then divides each sample by its mean value over time to remove effects due to differences in expression levels.
    #
    RAW_INTENSITIES: int = 0
    PHOTOBLEACHING: int = 1
    BACKGROUND_PHOTOBLEACHING_MEAN_IMAGE: int = 2
    PHOTOBLEACHING_MEAN_SAMPLE: int = 3
    BACKGROUND_PHOTOBLEACHING_MEAN_SAMPLE: int = 4
    BACKGROUND_PHOTOBLEACHING_MEAN_FILE: int = 5


class RCBBatchProcess(RCallback):
    VALID_EXTENSIONS: Tuple[str] = ('.tif', '.tiff', '.TIF', '.TIFF')
    OUTPUT_EXTENSION: str = '.tif'

    FFC_INPUT: str = '.'
    FFC_DARKFILE: str = ''
    FFC_FLATFILE: str = ''
    FFC_CROPDIMS: Tuple[int, int] = (512, 512)
    FFC_BG: str = 'mode'
    FFC_INPUTSUBSTR: str = ''
    FFC_FILESUFFIX: str = '_ffc_corrected'

    BATCH_MEASURE_SCRIPT: str = f"import sys\nsys.path.extend(['{PyJAMAS.folder.rstrip('/pyjamas')}'])\nfrom pyjamas.pjscore import PyJAMAS\na = PyJAMAS()\na.batch.cbMeasureBatch(parameters)"

    DIRS1_BM: str = '.'
    DIRS2_BM: str = ''
    ANALYZE_FLAG_BM: bool = True
    ANALYSIS_FILENAME_APPENDIX_BM: str = '_analysis'
    ANALYSIS_EXTENSION_BM: str = '.csv'
    SAVE_RESULTS: bool = True  # Save the analysis script in each folder in which an analysis flag is saved.
    RESULTS_FOLDER: str = './results'
    SCRIPT_EXTENSION_BM = '.py'
    SCRIPT_FILENAME_APPENDIX_BM: str = '_analysis_script' + PyJAMAS.notebook_extension
    INTENSITY_FLAG_BM: bool = True  # Run intensity section of the analysis/plots?
    IMAGE_EXTENSION_BM: str = '.tif'
    NORMALIZE_INTENSITY_FLAG_BM: int = normalization_modes.BACKGROUND_PHOTOBLEACHING_MEAN_IMAGE
    T_RES_BM: float = 30  # Time resolution in seconds.
    XY_RES_BM: float = 16 / (60 * 1.5)  # Spatial resolution in microns.
    # INDEX_TIME_ZERO_BM: Number of time points before treatment (e.g. number of images before wounding) if time zero is the time AFTER TREATMENT.
    # INDEX_TIME_ZERO_BM: Number of time points before treatment - 1 if time zero is the time BEFORE applying the treatment.
    INDEX_TIME_ZERO_BM: int = 4
    PLOT_FLAG_BM: bool = False  # Generate and display plots.
    GROUP_LABELS_BM: List[str] = ['group 1', 'group 2']
    ERR_STYLE_VALUE_BM: str = 'band'
    PLOT_STYLE_VALUE_BM: str = 'box'
    BRUSH_SZ_BM: int = 3

    COMPILE_DATA_FLAG_BM: bool = True  # Read all data and compile into DataFrames.
    NTP_BM: int = 308
    TIME_SHIFTS_BM = numpy.arange(-16, 17)

    # Keep time as the first measurements here.
    MEASUREMENTS_ABS = ['area (\u03BCm\u00B2)', 'perimeter (\u03BCm)', 'circularity (dimensionless)',
                        'raw pixel values interior', 'raw pixel values perimeter',
                        'image mean', 'image mode',
                        'normalized pixel values interior', 'normalized pixel values perimeter',
                        'std interior', 'std perimeter',
                        'normalized std interior', 'normalized std perimeter',
                        'heterogeneity interior', 'heterogeneity perimeter',
                        'total normalized pixel values interior', 'total normalized pixel values perimeter',
                        'scaled normalized pixel values interior', 'scaled normalized pixel values perimeter']
    MEASUREMENTS_PCTG = ['area (%)', 'perimeter (%)', 'circularity (%)', 'raw pixel values interior (%)',
                         'raw pixel values perimeter (%)', ]
    MEASUREMENTS_PCT_CHANGE = ['area (% change)', 'perimeter (% change)', 'circularity (% change)',
                               'normalized pixel values interior (% change)',
                               'normalized pixel values perimeter (% change)',
                               'total normalized pixel values interior (% change)',
                               'total normalized pixel values perimeter (% change)',
                               'heterogeneity interior (% change)',
                               'heterogeneity perimeter (% change)'
                               ]
    MEASUREMENT_LIST_BM = MEASUREMENTS_ABS + MEASUREMENTS_PCTG
    BOX_PLOT_MEASUREMENTS_BM = ['closure rate constant - area (1/min)', 'closure rate constant - perimeter (1/min)'] + MEASUREMENTS_PCT_CHANGE
    COLUMN_LIST_BM: List[str] = ['experimental group', 'experiment index', 'file name',
                                 'time (min)', ] + MEASUREMENT_LIST_BM + BOX_PLOT_MEASUREMENTS_BM
    ERRORBAR_BM = 'se'  # 'sd' for standard deviation, 'se' for standard error
    LINE_WIDTH_BM: int = 3

    def cbBatchProjectConcat(self, input_folder_name: Optional[str] = None, slice_str: Optional[str] = None,
                             output_file_name: Optional[str] = None,
                             projection_type: projection_types = projection_types.MAX,
                             wait_for_thread: bool = False) -> bool:
        """
        Project all the 3D images in a folder and concatenate all the maximum intensity projections into a new 3D image.

        A dialog will be opened if any parameters are set to None.

        :param input_folder_name: path to the folder that contains the images to project and concatenate.
        :param slice_str: slice indexes to use for projection (e.g. '0, 2-4, 12').
        :param output_file_name: path and file name to save the image resulting from projecting and concatenating the images in input_folder_name.
        :param projection_type: determines the type of projection to use (max, sum, etc.); value is one of rcbbatchprocess.projection_types.
        :param wait_for_thread: True if PyJAMAS must wait for the thread running this operation to complete, False otherwise.
        :return: True if the projected and concatenated image is saved, False otherwise.
        """
        # If not enough parameters, open dialog.
        if input_folder_name is None or input_folder_name == '' or input_folder_name is False or not os.path.exists(
                input_folder_name):  # When the menu option is clicked on, for some reason that I do not understand, the function is called with filename = False, which causes a bunch of problems.
            dialog = QtWidgets.QDialog()
            ui = BatchProjectConcatenateDialog()
            ui.setupUi(dialog)
            dialog.exec()
            dialog.show()
            # If the dialog was closed by pressing OK, then run the measurements.
            continue_flag = dialog.result() == QtWidgets.QDialog.DialogCode.Accepted
            theparameters = ui.parameters()
            theparameters['projection_type'] = projection_type

            dialog.close()

        # Otherwise, continue with supplied parameters
        else:
            theparameters = {'input_folder': input_folder_name,
                             'slice_list': slice_str,
                             'file_name': output_file_name,
                             'projection_type': projection_type}
            continue_flag = True

        if continue_flag:
            return_value = self.launch_thread(self.batch_project_concatenate,
                                              {'parameters': theparameters, 'progress': True, 'stop': True},
                                              finished_fn=self.finished_fn, stop_fn=self.stop_fn,
                                              progress_fn=self.progress_fn,
                                              wait_for_thread=wait_for_thread)

            if return_value:
                self.pjs.cwd = os.path.abspath(theparameters['input_folder'])

            return return_value
        else:
            return False

    def cbBatchResize(self, parameters: Optional[dict] = None) -> bool:
        """
        Resizes all images within a given folder or folder tree to a specified size and saves them to a new folder.

        :param parameters: dictionary with parameters, a dialog will open if the value is none.

            ``input_folder``: absolute path to the folder containing all images to be resized.
            ``save_folder``: absolute path to the folder that resized images will be saved to.
            ``im_size``: tuple containing the new image size in (rows, cols), eq. (height, width).
            ``recurr``: When set to true, the folder tree will be searched recursively for images to be resized.

        :return: True if resizing complete, False otherwise.
        """
        # Get folder name.
        continue_flag: bool = True

        if parameters is None or parameters is False:
            dialog = QtWidgets.QDialog()
            ui = BatchResizeDialog()
            ui.setupUi(dialog)

            dialog.exec()
            dialog.show()

            continue_flag = dialog.result() == QtWidgets.QDialog.DialogCode.Accepted

            if continue_flag:
                parameters = ui.parameters()

            dialog.close()

        if continue_flag:
            input_folder = parameters.get('input_folder')
            save_folder = parameters.get('save_folder')
            im_size = parameters.get('im_size')
            recurr = parameters.get('recurr')
            if save_folder == '':
                save_folder = input_folder + f"_resized_{im_size[0]}x{im_size[1]}"
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            self.batch_resize_images(input_folder, save_folder, im_size, recurr)
            self.pjs.statusbar.showMessage('Finished.')

            return True
        else:
            return False

    def batch_resize_images(self, input_folder: Optional[str] = None, save_folder: Optional[str] = None,
                            im_size: Optional[tuple] = (512, 512), recurr: Optional[bool] = True) -> bool:
        for subfile in os.listdir(input_folder):
            subfile_path = os.path.join(input_folder, subfile)
            if os.path.isdir(subfile_path) and recurr:
                if not os.path.exists(os.path.join(save_folder, subfile)):
                    os.mkdir(os.path.join(save_folder, subfile))
                self.batch_resize_images(subfile_path, os.path.join(save_folder, subfile), im_size, recurr)
            elif os.path.splitext(subfile_path)[1] == '.tif':
                img = sio.imread(subfile_path)
                if img.ndim == 3:
                    img = img[0, :, :]
                img = numpy.expand_dims(
                    st.resize(img, (im_size[0], im_size[1]), order=3, mode='constant', preserve_range=True), axis=-1)
                if numpy.unique(img).size == 2:  # Binary image, need to correct for interpolation
                    max_value = numpy.max(numpy.max(img))
                    img = numpy.round(numpy.divide(img, numpy.full(img.shape, max_value)))
                    img = numpy.multiply(img, numpy.full(img.shape, max_value))
                    img = numpy.maximum(img, numpy.zeros((im_size[0], im_size[1], 1)))
                sio.imsave(os.path.join(save_folder, subfile), img[:, :, 0], check_contrast=False)
            else:  # Not an acceptable image type or sub-directory
                continue

        return True

    def cbBatchFlatFieldCorrection(self, parameters: Optional[dict] = None) -> bool:
        """
        Flat field correction of all images within a given folder or folder tree.

        :param parameters: dictionary with parameters, a dialog will open if the value is none.

            ``input_folder``:
                absolute path to the folder containing all images to be resized.
            ``darkfield_file``:
                path to the file containing the image to correct for darkfield, '' or None of no darkfield correction.
            ``flatfield_file``:
                path to the file containing the image to correct for flatfield, '' or None of no flatfield correction.
            ``crop_dims``:
                tuple with (row, col) dimensions to crop the images before applying any corrections, None for no cropping.
            ``bg_mode``:
                background subtraction, None if no background subtraction, 'mode' if the background is the volume mode.
            ``input_substr``:
                substring to refine input files corrected, '' to correct all images.
            ``file_suffix``:
                suffix to add to the file name of the corrected images.

        :return: True if correction complete, False otherwise.
        """
        # Get folder name.
        continue_flag: bool = True

        if parameters is None or parameters is False:
            dialog = QtWidgets.QDialog()
            ui = BatchFlatFieldCorrectionDialog()
            ui.setupUi(dialog)

            dialog.exec()
            dialog.show()

            continue_flag = dialog.result() == QtWidgets.QDialog.DialogCode.Accepted

            if continue_flag:
                parameters = ui.parameters()

            dialog.close()

        if continue_flag:
            input_folder = parameters.get('input_folder')
            darkfield_file = parameters.get('darkfield_file')
            flatfield_file = parameters.get('flatfield_file')
            crop_dims = parameters.get('crop_dims')
            bg_mode = parameters.get('bg_mode')
            input_substr = parameters.get('input_substr')
            file_suffix = parameters.get('file_suffix')

            self.batch_correct_images(input_folder, darkfield_file, flatfield_file, crop_dims, bg_mode, input_substr,
                                      file_suffix)
            self.pjs.statusbar.showMessage('Finished.')

            return True
        else:
            return False

    def batch_correct_images(self, input_folder: Optional[str] = None, darkfield_file: Optional[str] = None,
                             flatfield_file: Optional[str] = None, crop_dims: Optional[tuple] = FFC_CROPDIMS,
                             bg_mode: Optional[str] = None, input_substr: Optional[str] = FFC_INPUTSUBSTR,
                             file_suffix: Optional[str] = FFC_FILESUFFIX):
        ffc_corr: bool = False
        dark_corr: bool = False
        crop_flag: bool = False
        crop_start_rows: int = -1
        crop_start_cols: int = -1
        nrows: int = -1  # number of rows to crop
        ncols: int = -1  # number of cols to crop

        if os.path.exists(flatfield_file):
            ffc_img = rimutils.read_stack(flatfield_file)

            # The images to be corrected will be divided by this one. Thus, there should be no zeros.
            if numpy.any(numpy.isclose(ffc_img, 0.)):
                print(f'The image in {flatfield_file} contains zeros and therefore is not a valid flat field image.')
                return False

            ffc_corr = True

        if os.path.exists(darkfield_file):
            dark_img = rimutils.read_stack(darkfield_file).astype(numpy.float32)
            dark_corr = True

        if crop_dims is not None and len(crop_dims) == 2:
            nrows = crop_dims[0]
            ncols = crop_dims[1]
            if ffc_corr:
                crop_start_rows = int((ffc_img.shape[0] - nrows) / 2)
                crop_start_cols = int((ffc_img.shape[1] - ncols) / 2)
                ffc_img = ffc_img[crop_start_rows:crop_start_rows + nrows, crop_start_cols:crop_start_cols + ncols]
            if dark_corr:
                if crop_start_rows == -1:
                    crop_start_rows = int((dark_img.shape[0] - nrows) / 2)
                    crop_start_cols = int((dark_img.shape[1] - ncols) / 2)
                dark_img = dark_img[crop_start_rows:crop_start_rows + nrows, crop_start_cols:crop_start_cols + ncols]
            crop_flag = True
        else:
            crop_start_rows = 0
            crop_start_cols = 0

            if ffc_corr:
                nrows = ffc_img.shape[0]
                ncols = ffc_img.shape[1]
            elif dark_corr:
                nrows = dark_img.shape[0]
                ncols = dark_img.shape[1]
            else:
                print(f'Nothing to do.')
                return False

        if file_suffix == '':
            thenow = datetime.now()
            thesuffix = thenow.strftime(
                f"{thenow.year:04}{thenow.month:02}{thenow.day:02}_{thenow.hour:02}{thenow.minute:02}{thenow.second:02}")
            file_suffix = f'_{thesuffix}'

        file_list: List[str] = RUtils.extract_file_paths(input_folder, self.VALID_EXTENSIONS)

        for file_path in file_list:
            folder_name, full_file_name = os.path.split(file_path)
            file_name, ext = os.path.splitext(full_file_name)

            if input_substr in file_name:
                img = rimutils.read_stack(file_path)

                if crop_flag and crop_start_rows == -1:
                    crop_start_rows = int((img.shape[1] - nrows) / 2)
                    crop_start_cols = int((img.shape[2] - ncols) / 2)

                new_img: numpy.ndarray = numpy.empty((img.shape[0], nrows, ncols), dtype=float)

                for anindex, aslice in enumerate(img):
                    tmp_img: numpy.ndarray = aslice.copy()

                    if crop_flag and nrows < tmp_img.shape[0] and ncols < tmp_img.shape[1]:
                        tmp_img = tmp_img[crop_start_rows:crop_start_rows + nrows,
                                  crop_start_cols:crop_start_cols + ncols]

                    if dark_corr:
                        tmp_img = numpy.clip(tmp_img - dark_img, 0, None)

                    if ffc_corr:
                        tmp_img = tmp_img / ffc_img

                        ## This code here deals with nan's. We are not using this because the function returns an
                        ## error if the ffc_image contains 0s.
                        # # need to deal with pixels that are nan after being divided by zero ... usic cubic interpolation.
                        # # this should not run if the ffc image does not have zero values.
                        # x = numpy.arange(0, tmp_img.shape[1])
                        # y = numpy.arange(0, tmp_img.shape[0])
                        # xx, yy = numpy.meshgrid(x, y)
                        # # mask invalid values
                        # masked_img = numpy.ma.masked_invalid(tmp_img)
                        #
                        # # get coordinates with valid/invalid values
                        # x_invalid = xx[masked_img.mask]
                        #
                        # if x_invalid.size > 0:
                        #     print(
                        #         f'WARNING: {x_invalid.size} pixels divided by zero in flat field correction of slice {anindex} in {afile}.')
                        #
                        #     y_invalid = yy[masked_img.mask]
                        #     x_valid = xx[~masked_img.mask]
                        #     y_valid = yy[~masked_img.mask]
                        #
                        #     valid_array = masked_img.data[~masked_img.mask]
                        #
                        #     interp_values = interpolate.griddata((x_valid, y_valid), valid_array.ravel(),
                        #                                          (x_invalid, y_invalid), method='cubic')
                        #
                        #     tmp_img[y_invalid, x_invalid] = interp_values

                    new_img[anindex, :, :] = tmp_img.copy()

                match bg_mode:
                    case 'mode':
                        stack_mode = stats.mode(new_img.reshape(-1))[0]
                        new_img = numpy.clip(new_img - stack_mode, 0, None)

                corrected_file_path = os.path.join(folder_name, file_name+file_suffix+ext)
                rimutils.write_stack(corrected_file_path, new_img.astype(numpy.uint16))

        return True

    @classmethod
    def _default_batchflatfield_parameters(cls) -> dict:
        return {
            'input_folder': RCBBatchProcess.FFC_INPUT,
            'darkfield_file': RCBBatchProcess.FFC_DARKFILE,
            'flatfield_file': RCBBatchProcess.FFC_FLATFILE,
            'crop_dims': RCBBatchProcess.FFC_CROPDIMS,
            'bg_mode': RCBBatchProcess.FFC_BG,
            'input_substr': RCBBatchProcess.FFC_INPUTSUBSTR,
            'file_suffix': RCBBatchProcess.FFC_FILESUFFIX
        }


    def cbMeasureBatch(self, parameters: Optional[dict] = None) -> bool:
        """
        Measures image data sets and produces plots and CSV files that combine all the data. A Python script to reproduce the analysis, and a Jupyter notebook to reproduce the analysis and generate the plots can also be generated.

        :param parameters: dictionary containing measurement parameters; a dialog will open if the value is set to None; dictionary keys are:

            ``folders``:
                paths to the folders containing each of the data sets to compared, formatted as a list of strings; each folder will contain subfolders, each with different images and annotation files
            ``names``:
                labels for the datasets to be compared, formatted as a list of strings
            ``t_res``:
                time resolution in seconds, formatted as a float
            ``index_time_zero``:
                the slice index in timeseries corresponding to time zero, formatted as an integer
            ``xy_res``:
                spatial resolution in microns, formatted as float
            ``brush_sz``:
                size of the brush used to quantify intensity under polyline annotations, formatted as an integer
            ``intensity_flag``:
                determines whether pixel values (and not only morphological features) are measured, set to True or False
            ``image_extension``:
                extension of the images to be loaded for intensity analysis, formatted a string
            ``normalize_intensity_flag``:
                determines the procedure to normalize pixel values in time sequences, set to one of pjs.batch.normalization_modes:
                    ``RAW_INTENSITIES`` – no normalization
                    ``PHOTOBLEACHING`` – divide by the image mean (the photobleaching normalization factor is 1 for first image in the sequence)
                    ``BACKGROUND_PHOTOBLEACHING_MEAN_IMAGE`` - subtract the image mode and divide by the image mean (the photobleaching normalization factor is 1 for first image in the sequence)
                    ``PHOTOBLEACHING_MEAN_SAMPLE`` - detect the sample and divide by the sample mean (the photobleaching normalization factor is 1 for first image in the sequence)
                    ``BACKGROUND_PHOTOBLEACHING_MEAN_SAMPLE`` - detect the sample, subtract the sample mode and divide by the sample mean (the photobleaching normalization factor is 1 for first image in the sequence)
                    ``BACKGROUND_PHOTOBLEACHING_MEAN_FILE`` - use a file with either a single slice or as many slices as the original image, and named as the original image with PyJAMAS.backgroundimage_extension (the photobleaching normalization factor is 1 for first image in the sequence).
            ``analysis_filename_appendix``:
                suffix to be added to the name of the csv file used to save the analysis results, formatted as a string
            ``analysis_extension``:
                extension of the csv file used to save analysis results, formatted as a string
            ``err_style_value``:
                determines how error is displayed in plots averaging multiple data sets, set to either ‘band’ or ‘bar’
            ``plot_style_value``:
                determines how distributions are displayed, set to either ‘box’ or ‘violin’
            ``analyze_flag``:
                determines whether annotations are measured, or if previously stored measurement values are used for plotting (faster), set to True or False
            ``save_results``:
                determines whether overall results are saved (in a csv file) and a Jupyter notebook to reproduce analysis and plot produced, set to True or False
            ``results_folder``:
                path to the folder where overall quantification results, notebook, and plots will be saved, formatted as a string
            ``script_filename_appendix``:
                appendix to be added to the name of a file containing a Jupyter notebook that can be used to reproduce the analysis results and plots, formatted as a string
            ``plot_flag``:
                determines if plots are generated (slower) or not, set to True or False

        :return: True if the measurements complete, False otherwise.
        """
        continue_flag: bool = True

        if parameters is None or parameters is False:
            dialog = QtWidgets.QDialog()
            ui = BatchMeasureDialog()
            ui.setupUi(dialog)

            dialog.exec()
            dialog.show()

            continue_flag = dialog.result() == QtWidgets.QDialog.DialogCode.Accepted

            if continue_flag:
                parameters = ui.parameters()

            dialog.close()

        if continue_flag:
            return self.batch_measure(parameters)

        else:
            return False

    def batch_project_concatenate(self, parameters: dict, progress_signal: ThreadSignals,
                                  stop_signal: ThreadSignals) -> bool:
        folder_name: str = parameters.get('input_folder')
        slices: str = parameters.get('slice_list')
        file_name: str = parameters.get('file_name')
        projection_type: projection_types = parameters.get('projection_type', projection_types.MAX)

        # Projection function is determined by the type of projection.
        if projection_type == projection_types.MAX:
            proj_fn = rimutils.mip
        elif projection_type == projection_types.SUM:
            proj_fn = rimutils.sip

        if file_name[-4:] != RCBBatchProcess.OUTPUT_EXTENSION:
            file_name = RUtils.set_extension(file_name, RCBBatchProcess.OUTPUT_EXTENSION)

        if not os.path.exists(folder_name):
            if stop_signal is not None:
                stop_signal.emit('Input folder does not exist!')
            return False

        file_list: List[str] = os.listdir(folder_name)
        file_list.sort(key=RUtils.natural_sort)

        n_files: int = len(file_list)

        projected_image: numpy.ndarray = None
        projected_array: numpy.ndarray = None

        for ind, thefile in enumerate(file_list):
            _, extension = os.path.splitext(thefile)

            if extension.lower() in RCBBatchProcess.VALID_EXTENSIONS:
                theimage: numpy.ndarray = rimutils.read_stack(os.path.join(folder_name, thefile))

                if slices == '' or not slices:
                    projected_image = numpy.expand_dims(proj_fn(theimage), axis=0)
                else:
                    slice_list: List[int] = RUtils.parse_range_list(slices)
                    projected_image = numpy.expand_dims(proj_fn(theimage[slice_list]), axis=0)

                if projected_array is not None:
                    projected_array = numpy.concatenate((projected_array, projected_image), axis=0)

                else:
                    projected_array = projected_image.copy()

            if progress_signal is not None:
                progress_signal.emit(int((100 * (ind + 1)) / n_files))

        # Now write the file.
        rimutils.write_stack(os.path.join(folder_name, file_name), projected_array)

        return True

    def batch_measure(self, parameters: dict) -> bool:
        self._read_parameters(parameters)

        if self.analyze_flag:
            self._measure_data(parameters)

        if self.compile_data_flag:
            all_data = self._combine_measured_data()

            if self.plot_flag:
                self._plot_data(all_data)

            if self.save_results:
                data_path: str = self._save_data(all_data, parameters)
                self._save_notebook(parameters, data_path)

        self.pjs.statusbar.showMessage(f"Analysis results stored in {parameters.get('results_folder')}.")

        return True

    @classmethod
    def _default_batchmeasure_parameters(cls) -> dict:
        return {
            'folders': [RCBBatchProcess.DIRS1_BM, RCBBatchProcess.DIRS2_BM],
            'analyze_flag': RCBBatchProcess.ANALYZE_FLAG_BM,
            'analysis_filename_appendix': RCBBatchProcess.ANALYSIS_FILENAME_APPENDIX_BM,
            'analysis_extension': RCBBatchProcess.ANALYSIS_EXTENSION_BM,
            'save_results': RCBBatchProcess.SAVE_RESULTS,
            'results_folder': RCBBatchProcess.RESULTS_FOLDER,
            'script_filename_appendix': RCBBatchProcess.SCRIPT_FILENAME_APPENDIX_BM,
            'intensity_flag': RCBBatchProcess.INTENSITY_FLAG_BM,
            'image_extension': RCBBatchProcess.IMAGE_EXTENSION_BM,
            'normalize_intensity_flag': RCBBatchProcess.NORMALIZE_INTENSITY_FLAG_BM,
            't_res': RCBBatchProcess.T_RES_BM,
            'xy_res': RCBBatchProcess.XY_RES_BM,
            'index_time_zero': RCBBatchProcess.INDEX_TIME_ZERO_BM,
            'plot_flag': RCBBatchProcess.PLOT_FLAG_BM,
            'names': RCBBatchProcess.GROUP_LABELS_BM,
            'err_style_value': RCBBatchProcess.ERR_STYLE_VALUE_BM,
            'plot_style_value': RCBBatchProcess.PLOT_STYLE_VALUE_BM,
            'brush_sz': RCBBatchProcess.BRUSH_SZ_BM,
        }

    def _read_parameters(self, parameters: dict) -> bool:
        self.folders: List[str] = parameters.get('folders', [RCBBatchProcess.DIRS1_BM, RCBBatchProcess.DIRS2_BM])

        if self.folders[0] == '' or not os.path.isdir(self.folders[0]):
            self.pjs.statusbar.showMessage(f'{self.folders[0]} is not a folder.')
            return False

        self.dirs: List[List[str]] = [RUtils.extract_file_paths(self.folders[0],
                                                                [PyJAMAS.data_extension, PyJAMAS.matlab_extension])]

        # This enables analysis of a single set of folders.
        for this_folder in self.folders[1:]:
            if this_folder != '' and os.path.isdir(this_folder):
                self.dirs.append(RUtils.extract_file_paths(this_folder,
                                                           [PyJAMAS.data_extension, PyJAMAS.matlab_extension]))
            else:
                self.pjs.statusbar.showMessage(f'Skipping {this_folder} as it is not a folder.')

        self.analyze_flag: bool = parameters.get('analyze_flag', RCBBatchProcess.ANALYZE_FLAG_BM)
        self.analysis_filename_appendix: str = parameters.get('analysis_filename_appendix',
                                                              RCBBatchProcess.ANALYSIS_FILENAME_APPENDIX_BM)
        self.analysis_extension: str = parameters.get('analysis_extension', RCBBatchProcess.ANALYSIS_EXTENSION_BM)
        self.save_results: bool = parameters.get('save_results',
                                                 RCBBatchProcess.SAVE_RESULTS)  # Save the analysis script in each folder in which an analysis flag is saved.
        self.results_folder: str = parameters.get('results_folder',
                                                  RCBBatchProcess.RESULTS_FOLDER)  # Save the analysis script in each folder in which an analysis flag is saved.
        # results_folder cannot be empty.
        if self.results_folder == '':
            self.results_folder = RCBBatchProcess.RESULTS_FOLDER

        self.script_filename_appendix: str = parameters.get('script_filename_appendix',
                                                            RCBBatchProcess.SCRIPT_FILENAME_APPENDIX_BM)
        self.intensity_flag: bool = parameters.get('intensity_flag',
                                                   RCBBatchProcess.INTENSITY_FLAG_BM)  # Run intensity section of the analysis/plots?
        self.image_extension: str = parameters.get('image_extension', RCBBatchProcess.IMAGE_EXTENSION_BM)
        self.normalize_intensity_flag: int = parameters.get('normalize_intensity_flag',
                                                            RCBBatchProcess.NORMALIZE_INTENSITY_FLAG_BM)
        self.t_res: float = parameters.get('t_res', RCBBatchProcess.T_RES_BM)  # Time resolution in seconds.
        self.xy_res: float = parameters.get('xy_res', RCBBatchProcess.XY_RES_BM)  # Spatial resolution in microns.
        self.index_time_zero: int = parameters.get('index_time_zero',
                                                   RCBBatchProcess.INDEX_TIME_ZERO_BM)  # Number of time points before treatment (e.g. number of images before wounding).

        self.plot_flag: bool = parameters.get('plot_flag', RCBBatchProcess.PLOT_FLAG_BM)  # Generate and display plots.
        self.group_labels: List[str] = parameters.get('names', RCBBatchProcess.GROUP_LABELS_BM)

        self.err_style_value: str = parameters.get('err_style_value', RCBBatchProcess.ERR_STYLE_VALUE_BM)
        self.plot_style_value: str = parameters.get('plot_style_value', RCBBatchProcess.PLOT_STYLE_VALUE_BM)
        self.brush_sz: int = parameters.get('brush_sz', RCBBatchProcess.BRUSH_SZ_BM)

        self.compile_data_flag: bool = RCBBatchProcess.COMPILE_DATA_FLAG_BM  # Read all data and compile into DataFrames.
        self.ntp: int = RCBBatchProcess.NTP_BM
        self.t = (numpy.arange(0, self.ntp) - self.index_time_zero) * self.t_res / 60.  # Time in minutes.
        self.time_shifts = RCBBatchProcess.TIME_SHIFTS_BM
        self.measurement_list = RCBBatchProcess.MEASUREMENT_LIST_BM
        self.box_plot_measurements = RCBBatchProcess.BOX_PLOT_MEASUREMENTS_BM
        self.column_list: List[str] = RCBBatchProcess.COLUMN_LIST_BM
        self.errorbar: str = parameters.get('errorbar', RCBBatchProcess.ERRORBAR_BM)
        self.line_width: int = RCBBatchProcess.LINE_WIDTH_BM

        return True

    def _measure_data(self, parameters: dict) -> bool:
        prev_image: numpy.ndarray = self.pjs.slices.copy()
        prev_brush_sz: int = self.pjs.brush_size
        prev_wd: str = os.getcwd()
        prev_polylines = self.pjs.polylines
        prev_polyline_ids = self.pjs.polyline_ids
        prev_fiducials = self.pjs.fiducials

        self.pjs.options.cbSetBrushSize(self.brush_sz)

        if self.intensity_flag:
            if self.normalize_intensity_flag in [normalization_modes.RAW_INTENSITIES,
                                                 normalization_modes.PHOTOBLEACHING,
                                                 normalization_modes.BACKGROUND_PHOTOBLEACHING_MEAN_IMAGE]:
                measurement_flags = {'area': True, 'perimeter': True, 'pixels': True, 'image': True,
                                     'sample': False}
            elif self.normalize_intensity_flag in [normalization_modes.PHOTOBLEACHING_MEAN_SAMPLE,
                                                   normalization_modes.BACKGROUND_PHOTOBLEACHING_MEAN_SAMPLE]:
                measurement_flags = {'area': True, 'perimeter': True, 'pixels': True, 'image': False,
                                     'sample': True}
            elif self.normalize_intensity_flag == normalization_modes.BACKGROUND_PHOTOBLEACHING_MEAN_FILE:
                measurement_flags = {'area': True, 'perimeter': True, 'pixels': True, 'image': False,
                                     'sample': False}
            image_file_path = None
        else:
            measurement_flags = {'area': True, 'perimeter': True, 'pixels': False, 'image': False,
                                 'sample': False}

            image_file_path = ''

        for folder_list in self.dirs:
            for ii, annotations_file in enumerate(folder_list):
                print(f"Analyzing movie {ii + 1}/{len(folder_list)} ... ", end="")

                file_path, full_file_name = os.path.split(annotations_file)
                file_name, ext = os.path.splitext(full_file_name)

                # Create analysis file path.
                full_analysis_file_name = os.path.join(file_path,
                                                       file_name + self.analysis_filename_appendix + self.analysis_extension)
                full_script_file_name = os.path.join(file_path,
                                                     file_name + self.script_filename_appendix)

                # Figure out image file name.
                if self.intensity_flag:
                    _, folder_name = os.path.split(file_path)
                    image_file_path = os.path.join(file_path, folder_name + self.image_extension)
                    if not os.path.exists(image_file_path):
                        image_file_path = os.path.join(file_path, folder_name + '_488' + self.image_extension)
                        if not os.path.exists(image_file_path):
                            image_file_path = os.path.join(file_path, folder_name + '_' + self.image_extension)
                            if not os.path.exists(image_file_path):
                                thefiles = [f for f in os.listdir(file_path) if f.endswith(self.image_extension)]
                                image_file_path = os.path.join(file_path, thefiles[0]) if thefiles else ''

                # Load annotation file (and image if intensity measurements are happening).
                if os.path.isfile(annotations_file) and ext == PyJAMAS.data_extension:
                    self.pjs.io.cbLoadAnnotations([annotations_file], image_file=image_file_path)
                elif os.path.isfile(annotations_file) and ext == PyJAMAS.matlab_extension:
                    self.pjs.io.cbImportSIESTAAnnotations([annotations_file], image_file=image_file_path)
                else:
                    continue

                # Find indexes of first and last slice with polylines.
                first: int = 0

                for slice in self.pjs.polylines:
                    if slice == []:
                        first += 1
                    else:
                        break

                last: int = len(self.pjs.polylines) - 1

                for slice in self.pjs.polylines[-1::-1]:
                    if slice == []:
                        last -= 1
                    else:
                        break

                self.pjs.measurements.cbMeasurePoly(first + 1, last + 1, measurement_flags, full_analysis_file_name)

                # Analysis script saved here so that we know where each analysis result file came from.
                if self.save_results:
                    with open(full_script_file_name, "w") as f:
                        print(f"parameters = {str(parameters)}\n{RCBBatchProcess.BATCH_MEASURE_SCRIPT}", file=f)

                print("done!")

        # Restore GUI to its state before beginning batch analysis.
        self.pjs.io.cbLoadArray(prev_image)
        self.pjs.options.cbSetCWD(prev_wd)
        self.pjs.options.cbSetBrushSize(prev_brush_sz)
        self.pjs.polylines = prev_polylines
        self.pjs.polyline_ids = prev_polyline_ids
        self.pjs.fiducials = prev_fiducials

        return True

    def _combine_measured_data(self) -> pd.DataFrame:
        n_files_to_analyze = sum([len(this_dir) for this_dir in self.dirs])

        # all_data is the DataFrame in which all the data will be compiled.
        # It contains one row per folder (i.e. annotation file) and time point.
        # Each row contains multiple columns with the values for different metrics in different folders and time points.
        all_data: pd.DataFrame = pd.DataFrame(
            numpy.nan * numpy.zeros((n_files_to_analyze * self.ntp, len(self.column_list))),
            columns=self.column_list)

        min_ind: int = 0
        max_ind: int = 0

        the_time_zero = self.index_time_zero if self.index_time_zero else 1  ## if the first image is the first time point, use one image to calculate % changes.

        for ii, annotations_file in enumerate(chain.from_iterable(self.dirs)):
            file_path, full_file_name = os.path.split(annotations_file)
            file_name, ext = os.path.splitext(full_file_name)

            # Create analysis file path.
            full_analysis_file_name = os.path.join(file_path,
                                                   file_name + self.analysis_filename_appendix + self.analysis_extension)

            msr_df = pd.read_csv(full_analysis_file_name, index_col=0)

            # Obtain the array of time points with measurements available.
            # The -1 here converts from slice numbers (starting at 1), into indices (starting at zero).
            index_list = numpy.asarray(msr_df.values[0] - 1, dtype=int)

            # min_ind and max_ind delimit the rows in all_data that correspond to the time points for the current folder.
            min_ind = max_ind
            max_ind = min_ind + self.ntp

            # Determine both the experimental group and the experiment index within that group.
            dirs_group = [annotations_file in this_dir for this_dir in self.dirs].index(True)
            all_data.iloc[min_ind:max_ind, all_data.columns.get_loc('experimental group')] = self.group_labels[dirs_group]  # this triggers a copy of a slice warning: all_data['experimental group'].iloc[min_ind:max_ind] = self.group_labels[dirs_group]
            all_data.iloc[min_ind:max_ind, all_data.columns.get_loc('experiment index')] = self.dirs[dirs_group].index(annotations_file)

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('time (min)')] = self.t[0:index_list.size]
            all_data.iloc[min_ind:max_ind, all_data.columns.get_loc('file name')] = full_analysis_file_name

            # REMEMBER: index_list is an ndarray with the relative indexes to use.
            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('area (\u03BCm\u00B2)')] = numpy.asarray(msr_df.loc['area_1']) * (
                    self.xy_res ** 2)

            areas = all_data['area (\u03BCm\u00B2)'].values[min_ind+index_list]
            times = all_data['time (min)'].values[min_ind+index_list]
            max_area_ind = numpy.argmax(areas)

            # Fits exponential from maximum (if there are at least 3 time points).
            if areas.size >= 3 and areas[max_area_ind:].size >= 3:
                x, flag = leastsq(RUtils.residuals, (areas[max_area_ind], 1.0),
                                  args=(RUtils.func_exp_2params, areas[max_area_ind:], times[max_area_ind:]))

                # Verify fits.
                # plt.figure()
                # ax = sns.lineplot(x=times, y=areas)
                # ax = sns.lineplot(x=times[max_area_ind:], y=RUtils.func_exp_2params(times[max_area_ind:], x), marker="o")
                # r, p = numpy.corrcoef(areas[max_area_ind:], RUtils.func_exp_2params(times[max_area_ind:], x))

                if flag == 1:
                    all_data.iloc[min_ind+index_list, all_data.columns.get_loc('closure rate constant - area (1/min)')] = 1./x[1]
                else:
                    print(f"WARNING: could not fit an exponential to the area data for {annotations_file}.")
                    all_data.iloc[
                        min_ind + index_list, all_data.columns.get_loc('closure rate constant - area (1/min)')] = numpy.nan
            else:
                self.box_plot_measurements.pop(self.box_plot_measurements.index('closure rate constant - area (1/min)'))

            mean_area_beg: float = numpy.nanmean(numpy.asarray(msr_df.loc['area_1'])[0:the_time_zero])
            mean_area_end: float = numpy.nanmean(numpy.asarray(msr_df.loc['area_1'][-the_time_zero:]))

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('area (%)')] = 100. * numpy.asarray(
                msr_df.loc['area_1']) / mean_area_beg

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('area (% change)')] = 100. * (
                    mean_area_end - mean_area_beg) / mean_area_beg

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('perimeter (\u03BCm)')] = numpy.asarray(
                msr_df.loc['perimeter_1']) * self.xy_res

            perimeters = all_data['perimeter (\u03BCm)'].values[min_ind+index_list]
            max_perimeter_ind = numpy.argmax(perimeters)

            # Fits exponential from maximum (if there are at least 3 time points).
            if perimeters.size >= 3 and perimeters[max_perimeter_ind:].size >= 3:
                x, flag = leastsq(RUtils.residuals, (perimeters[max_perimeter_ind], 1.0),
                                  args=(RUtils.func_exp_2params, perimeters[max_perimeter_ind:], times[max_perimeter_ind:]))

                # Verify fits.
                # plt.figure()
                # ax = sns.lineplot(x=times, y=areas)
                # ax = sns.lineplot(x=times[max_area_ind:], y=RUtils.func_exp_2params(times[max_area_ind:], x), marker="o")
                # r, p = numpy.corrcoef(areas[max_area_ind:], RUtils.func_exp_2params(times[max_area_ind:], x))
                if flag == 1:
                    all_data.iloc[min_ind+index_list, all_data.columns.get_loc('closure rate constant - perimeter (1/min)')] = 1./x[1]
                else:
                    print(f"WARNING: could not fit an exponential to the perimeter data for {annotations_file}.")
                    all_data.iloc[min_ind + index_list, all_data.columns.get_loc(
                        'closure rate constant - perimeter (1/min)')] = numpy.nan
            else:
                self.box_plot_measurements.pop(self.box_plot_measurements.index('closure rate constant - perimeter (1/min)'))

            mean_perimeter_beg: float = numpy.nanmean(numpy.asarray(msr_df.loc['perimeter_1'])[0:the_time_zero])
            mean_perimeter_end: float = numpy.nanmean(numpy.asarray(msr_df.loc['perimeter_1'][-the_time_zero:]))

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('perimeter (%)')] = 100. * numpy.asarray(
                msr_df.loc['perimeter_1']) / mean_perimeter_beg
            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('perimeter (% change)')] = 100. * (
                    mean_perimeter_end - mean_perimeter_beg) / mean_perimeter_beg

            mean_circularity_beg: float = numpy.nanmean(
                numpy.asarray(msr_df.loc['area_1'])[0:the_time_zero] * (self.xy_res ** 2) / (
                        numpy.asarray(msr_df.loc['perimeter_1'])[
                        0:the_time_zero] * self.xy_res))
            mean_circularity_end: float = numpy.nanmean(
                numpy.asarray(msr_df.loc['area_1'])[-the_time_zero] * (self.xy_res ** 2) / (
                        numpy.asarray(msr_df.loc['perimeter_1'])[
                        -the_time_zero:] * self.xy_res))

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('circularity (%)')] = 100. * (
                    all_data['area (\u03BCm\u00B2)'].values[
                        min_ind+index_list] /
                    all_data['perimeter (\u03BCm)'].values[
                        min_ind+index_list]) / mean_circularity_beg

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('circularity (% change)')] = 100. * (
                    mean_circularity_end - mean_circularity_beg) / mean_circularity_beg

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('raw pixel values interior')] = numpy.asarray(
                msr_df.loc['pixel_values_interior_1'])
            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('raw pixel values interior (%)')] = 100. * numpy.asarray(
                msr_df.loc['pixel_values_interior_1']) / numpy.nanmean(
                numpy.asarray(msr_df.loc['pixel_values_interior_1'])[0:the_time_zero])

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('raw pixel values perimeter')] = numpy.asarray(
                msr_df.loc['pixel_values_perimeter_1'])
            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('raw pixel values perimeter (%)')] = 100. * numpy.asarray(
                msr_df.loc['pixel_values_perimeter_1']) / numpy.nanmean(
                numpy.asarray(msr_df.loc['pixel_values_perimeter_1'])[0:the_time_zero])

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('std interior')] = numpy.asarray(
                msr_df.loc['std_interior_1'])
            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('std perimeter')] = numpy.asarray(
                msr_df.loc['std_perimeter_1'])

            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('image mean')] = numpy.asarray(msr_df.loc['image_mean'])
            all_data.iloc[min_ind+index_list, all_data.columns.get_loc('image mode')] = numpy.asarray(msr_df.loc['image_mode'])

        all_data['circularity (dimensionless)'] = 4 * numpy.pi * all_data['area (\u03BCm\u00B2)'].values / numpy.square(
            all_data['perimeter (\u03BCm)'].values)

        if self.normalize_intensity_flag == normalization_modes.RAW_INTENSITIES:
            all_data['normalized pixel values interior'] = all_data['raw pixel values interior'].values
            all_data['normalized pixel values perimeter'] = all_data['raw pixel values perimeter'].values
            all_data['normalized std interior'] = all_data['std interior'].values
            all_data['normalized std perimeter'] = all_data['std perimeter'].values

        elif self.normalize_intensity_flag in [normalization_modes.PHOTOBLEACHING,
                                               normalization_modes.PHOTOBLEACHING_MEAN_SAMPLE]:
            first_mean_values = all_data.groupby(['experimental group', 'experiment index'], as_index=False, sort=False).first()['image mean'].values.repeat(self.ntp)  # use to multiply values after photobleaching correction, such that the photobleaching correction factor == 1 for the first image in the sequence.
            all_data['normalized pixel values interior'] = first_mean_values * all_data['raw pixel values interior'].values / all_data[
                'image mean'].values
            all_data['normalized pixel values perimeter'] = first_mean_values * all_data['raw pixel values perimeter'] / all_data[
                'image mean'].values
            all_data['normalized std interior'] = first_mean_values * all_data['std interior'].values / all_data['image mean'].values
            all_data['normalized std perimeter'] = first_mean_values * all_data['std perimeter'].values / all_data['image mean'].values

        elif self.normalize_intensity_flag in [normalization_modes.BACKGROUND_PHOTOBLEACHING_MEAN_IMAGE,
                                               normalization_modes.BACKGROUND_PHOTOBLEACHING_MEAN_SAMPLE,
                                               normalization_modes.BACKGROUND_PHOTOBLEACHING_MEAN_FILE]:
            first_mean_values = all_data.groupby(['experimental group', 'experiment index'], as_index=False, sort=False).first()['image mean'].values.repeat(self.ntp)  # use to multiply values after photobleaching correction, such that the photobleaching correction factor == 1 for the first image in the sequence.
            all_data['normalized pixel values interior'] = first_mean_values * (all_data['raw pixel values interior'].values - all_data[
                'image mode'].values) / all_data['image mean'].values
            all_data['normalized pixel values perimeter'] = first_mean_values * (all_data['raw pixel values perimeter'] - all_data[
                'image mode'].values) / all_data['image mean'].values
            # Clip intensities at zero. Use .loc here to avoid "SettingWithCopyWarning" -> see https://www.dataquest.io/blog/settingwithcopywarning.
            all_data.loc[all_data['normalized pixel values interior'] < 0, 'normalized pixel values interior'] = 0.0
            all_data.loc[all_data['normalized pixel values perimeter'] < 0, 'normalized pixel values perimeter'] = 0.0
            print(f"WARNING: some intensities were clipped at zero.")

            all_data['normalized std interior'] = first_mean_values * all_data['std interior'].values / all_data['image mean'].values
            all_data['normalized std perimeter'] = first_mean_values * all_data['std perimeter'].values / all_data['image mean'].values

        all_data['total normalized pixel values perimeter'] = all_data['normalized pixel values perimeter'] * all_data[
            'perimeter (\u03BCm)'].values
        all_data['total normalized pixel values interior'] = all_data['normalized pixel values interior'] * all_data[
            'area (\u03BCm\u00B2)'].values

        all_data['heterogeneity interior'] = all_data['normalized std interior'].values / all_data[
            'normalized pixel values interior'].values
        all_data['heterogeneity perimeter'] = all_data['normalized std perimeter'].values / all_data[
            'normalized pixel values perimeter'].values

        # Percent change in pixel values calculated here.
        thedata = all_data.groupby(['experimental group', 'experiment index'], as_index=False, sort=False)

        for a_name, a_group in thedata:
            the_group = a_group[a_group['time (min)'].notna()]

            values_interior = the_group['normalized pixel values interior'].values
            mean_interior_beg: float = numpy.nanmean(values_interior[0:the_time_zero])
            mean_interior_end: float = numpy.nanmean(values_interior[-the_time_zero:])

            values_perimeter = the_group['normalized pixel values perimeter'].values
            mean_perimeter_beg: float = numpy.nanmean(values_perimeter[0:the_time_zero])
            mean_perimeter_end: float = numpy.nanmean(values_perimeter[-the_time_zero:])

            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data['time (min)'].notna()), 'normalized pixel values interior (% change)'] = 100. * (
                    mean_interior_end - mean_interior_beg) / mean_interior_beg
            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data['time (min)'].notna()), 'normalized pixel values perimeter (% change)'] = 100. * (
                    mean_perimeter_end - mean_perimeter_beg) / mean_perimeter_beg

            values_image = the_group['image mean'].values
            mean_image_beg: float = numpy.nanmean(values_image[0:the_time_zero])

            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data[
                    'time (min)'].notna()), 'scaled normalized pixel values interior'] = mean_image_beg * values_interior
            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data[
                    'time (min)'].notna()), 'scaled normalized pixel values perimeter'] = mean_image_beg * values_perimeter

            values_interior = the_group['total normalized pixel values interior'].values
            mean_interior_beg = numpy.nanmean(values_interior[0:the_time_zero])
            mean_interior_end = numpy.nanmean(values_interior[-the_time_zero:])

            values_perimeter = the_group['total normalized pixel values perimeter'].values
            mean_perimeter_beg = numpy.nanmean(values_perimeter[0:the_time_zero])
            mean_perimeter_end = numpy.nanmean(values_perimeter[-the_time_zero:])

            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data['time (min)'].notna()), 'total normalized pixel values interior (% change)'] = 100. * (
                    mean_interior_end - mean_interior_beg) / mean_interior_beg
            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data['time (min)'].notna()), 'total normalized pixel values perimeter (% change)'] = 100. * (
                    mean_perimeter_end - mean_perimeter_beg) / mean_perimeter_beg

            values_interior = the_group['heterogeneity interior'].values
            mean_interior_beg = numpy.nanmean(values_interior[0:the_time_zero])
            mean_interior_end = numpy.nanmean(values_interior[-the_time_zero:])

            values_perimeter = the_group['heterogeneity perimeter'].values
            mean_perimeter_beg = numpy.nanmean(values_perimeter[0:the_time_zero])
            mean_perimeter_end = numpy.nanmean(values_perimeter[-the_time_zero:])

            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data['time (min)'].notna()), 'heterogeneity interior (% change)'] = 100. * (
                    mean_interior_end - mean_interior_beg) / mean_interior_beg
            all_data.loc[(all_data['experimental group'] == a_name[0]) & (all_data['experiment index'] == a_name[1]) & (
                all_data['time (min)'].notna()), 'heterogeneity perimeter (% change)'] = 100. * (
                    mean_perimeter_end - mean_perimeter_beg) / mean_perimeter_beg

        return all_data

    def _plot_data(self, all_data: pd.DataFrame) -> bool:
        # We are going to plot A LOT. So we silence the warning that too many plots were opened.
        warnings.filterwarnings("ignore", message="More than 20 figures have been opened.")

        self._setup_plots()

        # Mean plots comparing two groups.
        n_groups: int = len(numpy.unique(all_data['experimental group']))
        a_palette = sns.color_palette('bright',
                                      n_groups)  # The "bright" colour palette will cycle for more than 10 colours.

        # todo: outward pointing ticks, axes units.
        for a_measurement in self.measurement_list:
            plt.figure()
            ax = sns.lineplot(x='time (min)', y=a_measurement, hue='experimental group', data=all_data,
                              errorbar=self.errorbar,
                              err_style=self.err_style_value, lw=self.line_width, palette=a_palette)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=self.group_labels, frameon=False)

            if not numpy.isinf(ax.dataLim.ymax):
                ax.set_ylim([min(ax.dataLim.ymin - abs(.05 * ax.dataLim.ymin), 0),
                             ax.dataLim.ymax + abs(.05 * ax.dataLim.ymax)])

            sns.despine()
            plt.show()

        # Plot individual curves for each of the folders.
        # All in the same plot.
        for a_measurement in self.measurement_list:
            plt.figure()
            ax = sns.lineplot(x='time (min)', y=a_measurement, hue='experimental group', data=all_data, estimator=None,
                              units='experiment index', lw=self.line_width, legend='full', palette=a_palette)
            if not numpy.isinf(ax.dataLim.ymax):
                ax.set_ylim([min(ax.dataLim.ymin - abs(.05 * ax.dataLim.ymin), 0),
                             ax.dataLim.ymax + abs(.05 * ax.dataLim.ymax)])
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[0:], labels=labels[0:], frameon=False)

            sns.despine()
            plt.show()

            # Each group independently, color-coding based on experiment index within the group.
            for a_group in self.group_labels:
                thedata = all_data[all_data['experimental group'] == a_group]

                if numpy.array(thedata).size == 0:
                    continue

                n_exp: int = len(numpy.unique(thedata['experiment index']).astype(int))

                plt.figure()
                ax = sns.lineplot(x='time (min)', y=a_measurement, hue='experiment index', data=thedata,
                                  estimator=None, units='experiment index', legend=False, lw=self.line_width,
                                  palette=sns.color_palette("husl",
                                                            n_exp))  # draw evenly-spaced colors in a circular color space
                if not numpy.isinf(ax.dataLim.ymax):
                    ax.set_ylim([min(ax.dataLim.ymin - abs(.05 * ax.dataLim.ymin), 0),
                                 ax.dataLim.ymax + abs(.05 * ax.dataLim.ymax)])
                ax.legend([str(x) for x in range(n_exp)], frameon=False, title=a_group)
                sns.despine()
                plt.show()

        # Box plots for metrics that summarize experiments with a number each (eg. area rate constant).
        # The rate constant is the same for all the time points of a given image, so we only take the first
        # value per image.
        # sort=False is important to preserve the order of the experimental groups, so that labels in the
        # plots below are correct.
        thedata = all_data.groupby(['experimental group', 'experiment index'], as_index=False, sort=False).first()

        for a_measurement in self.box_plot_measurements:
            plt.figure()

            if self.plot_style_value == 'box':
                # whis is the proportion of the IQR past the low and high quartiles to extend the plot whiskers.
                # Points outside this range will be identified as outliers (and error bars will ignore them).
                ax = sns.boxplot(x='experimental group', y=a_measurement, data=thedata, hue='experimental group',
                                 dodge=False,
                                 whis=100000, palette=a_palette)
            elif self.plot_style_value == 'violin':
                ax = sns.violinplot(x='experimental group', y=a_measurement, data=thedata, hue='experimental group',
                                    dodge=False)

            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=handles, frameon=False)

            ax = sns.stripplot(x='experimental group', y=a_measurement, data=thedata, color='k', alpha=0.75, size=6,
                               dodge=False,
                               jitter=0.05)
            if not numpy.isinf(ax.dataLim.ymax):
                ax.set_ylim([min(ax.dataLim.ymin - abs(.05 * ax.dataLim.ymin), 0),
                             ax.dataLim.ymax + abs(.05 * ax.dataLim.ymax)])

            sns.despine()
            plt.show()

        return True

    def _setup_plots(self) -> bool:
        sns.set()
        sns.set_style("white")

        plt.rcParams['font.size'] = 14
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['figure.figsize'] = [8, 6]
        # plt.xkcd()  ## xkcd plotting style :)

        return True

    def _save_data(self, all_data: pd.DataFrame, parameters: dict) -> str:
        # Returns the path where the data were saved.

        # Create folder self.results_folder if it does not exist.
        if not os.path.isdir(self.results_folder):
            os.mkdir(self.results_folder)

        # Create filename.
        thenow = datetime.now()
        filename = thenow.strftime(
            f"{thenow.year:04}{thenow.month:02}{thenow.day:02}_{thenow.hour:02}{thenow.minute:02}{thenow.second:02}")

        # This ensures paths that work in both Unix and Windows.
        filepath = self.results_folder + filename if self.results_folder.endswith(
            '/') else self.results_folder + '/' + filename

        # Save DataFrame to folder.
        all_data.to_csv(RUtils.set_extension(filepath + self.analysis_filename_appendix, self.analysis_extension))

        # Save Python script to folder.
        with open(RUtils.set_extension(filepath + self.script_filename_appendix, RCBBatchProcess.SCRIPT_EXTENSION_BM),
                  "w") as f:
            print(f"parameters = {str(parameters)}\n{RCBBatchProcess.BATCH_MEASURE_SCRIPT}", file=f)

        # Save all open figures.
        for index in plt.get_fignums():
            if plt.fignum_exists(index):
                plt.figure(index)
                plt.savefig(filepath + f"_{index:03}.svg")
                plt.savefig(filepath + f"_{index:03}.png")
                plt.close()

        return filepath

    def _save_notebook(self, parameters: dict, data_path: str) -> bool:
        nb: NotebookNode = self._save_widget_notebook_setup(data_path, parameters)
        nb.cells.extend(self._save_widget_notebook_meanplots(parameters).cells)
        nb.cells.extend(self._save_widget_notebook_singleplots().cells)
        nb.cells.extend(self._save_widget_notebook_boxplots().cells)

        nb['metadata'].update({'language_info': {'name': 'python'}})

        fname = RUtils.set_extension(data_path, PyJAMAS.notebook_extension)

        with open(fname, 'w', encoding="utf-8") as f:
            nbf.write(nb, f)

        return True

    def _save_notebook_setup(self, data_path: str, parameters: dict) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""# PyJAMAS notebook {data_path[data_path.rfind(os.sep) + 1:]}"""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        text = """We start by importing the packages necessary to run and plot the analysis:"""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"import matplotlib\n" \
               f"import matplotlib.pyplot as plt\n" \
               f"%matplotlib inline\n" \
               f"import numpy\n" \
               f"import pandas as pd\n" \
               f"import seaborn as sns"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        text = """Run the analysis (uncomment the code in this cell; otherwise see below to load analysis results from disk):"""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"# import sys\n" \
               f"# sys.path.extend(['{PyJAMAS.folder.rstrip('/pyjamas')}'])\n" \
               f"# from pyjamas.pjscore import PyJAMAS\n\n" \
               f"# a = PyJAMAS()\n\n" \
               f"# parameters = {str(parameters)}\n\n" \
               f"# a.batch.cbMeasureBatch(parameters)"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        text = """Or load analysis results from disk:"""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"all_data = pd.read_csv('{RUtils.set_extension(data_path + parameters.get('analysis_filename_appendix', RCBBatchProcess.ANALYSIS_FILENAME_APPENDIX_BM), parameters.get('analysis_extension', RCBBatchProcess.ANALYSIS_EXTENSION_BM))}')"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        text = """Plot results: first, set up plots."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"sns.set()\n" \
               f"sns.set_style(\"white\")\n" \
               f"plt.rcParams['font.size'] = 14\n" \
               f"plt.rcParams['font.weight'] = 'bold'\n" \
               f"plt.rcParams['xtick.labelsize'] = 16\n" \
               f"plt.rcParams['ytick.labelsize'] = 16\n" \
               f"plt.rcParams['axes.labelsize'] = 18\n" \
               f"plt.rcParams['axes.labelweight'] = 'bold'\n" \
               f"plt.rcParams['legend.fontsize'] = 14\n" \
               f"plt.rcParams['figure.figsize'] = [8, 6]\n" \
               f"# plt.xkcd()  # xkcd plotting style :)\n\n" \
               f"n_groups: int = len(numpy.unique(all_data['experimental group']))\n" \
               f"a_palette = sns.color_palette('bright', n_groups)"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb

    def _save_widget_notebook_setup(self, data_path: str, parameters: dict) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""# PyJAMAS notebook {data_path[data_path.rfind(os.sep) + 1:]}"""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        text = """We start by importing the packages necessary to run and plot the analysis. We also create lists that define what we will be plotting."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"import ipywidgets as widgets\n" \
               f"import matplotlib\n" \
               f"import matplotlib.pyplot as plt\n" \
               f"%matplotlib inline\n" \
               f"import numpy\n" \
               f"import pandas as pd\n" \
               f"import seaborn as sns"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        code = f"TIME_PLOTS = {self.measurement_list}\n" \
               f"BOX_PLOTS = {self.box_plot_measurements}"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        text = """Run the analysis (uncomment the code in this cell; otherwise see below to load analysis results from disk):"""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"# import sys\n" \
               f"# sys.path.extend(['{PyJAMAS.folder.rstrip('/pyjamas')}'])\n" \
               f"# from pyjamas.pjscore import PyJAMAS\n\n" \
               f"# a = PyJAMAS()\n\n" \
               f"# parameters = {str(parameters)}\n\n" \
               f"# a.batch.cbMeasureBatch(parameters)"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        text = """Or load analysis results from disk:"""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"all_data = pd.read_csv('{RUtils.set_extension(data_path + parameters.get('analysis_filename_appendix', RCBBatchProcess.ANALYSIS_FILENAME_APPENDIX_BM), parameters.get('analysis_extension', RCBBatchProcess.ANALYSIS_EXTENSION_BM))}')"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        text = """Plot results: first, set up plots."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"sns.set()\n" \
               f"sns.set_style(\"white\")\n" \
               f"plt.rcParams['font.size'] = 14\n" \
               f"plt.rcParams['font.weight'] = 'bold'\n" \
               f"plt.rcParams['xtick.labelsize'] = 16\n" \
               f"plt.rcParams['ytick.labelsize'] = 16\n" \
               f"plt.rcParams['axes.labelsize'] = 18\n" \
               f"plt.rcParams['axes.labelweight'] = 'bold'\n" \
               f"plt.rcParams['legend.fontsize'] = 14\n" \
               f"plt.rcParams['figure.figsize'] = [8, 6]\n" \
               f"# plt.xkcd()  # xkcd plotting style :)\n\n" \
               f"n_groups: int = len(numpy.unique(all_data['experimental group']))\n" \
               f"a_palette = sns.color_palette('bright', n_groups)"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb

    def _save_notebook_meanplots(self, parameters: dict) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""## Means plots comparing two groups.\n""" \
               f"""You may use *plt.savefig("file_name.ext")* after the code that creates a figure to save the figure. *ext* can be png, svg, etc ..."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        for a_measurement in self.measurement_list:
            text = f"""{a_measurement} *vs*. time (min):"""
            nb['cells'].append(nbf.v4.new_markdown_cell(text))

            code = f"ax = sns.lineplot(x='time (min)', y='{a_measurement}', hue='experimental group', data=all_data, errorbar='{parameters.get('errorbar', RCBBatchProcess.ERRORBAR_BM)}', err_style='{parameters.get('err_style_value', RCBBatchProcess.ERR_STYLE_VALUE_BM)}', lw={RCBBatchProcess.LINE_WIDTH_BM}, palette=a_palette)\n" \
                   f"handles, _ = ax.get_legend_handles_labels()\n" \
                   f"ax.legend(handles=handles, labels={parameters.get('names')}, frameon=False)\n" \
                   f"sns.despine()"
            nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb

    def _save_widget_notebook_meanplots(self, parameters: dict) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""## Means plots comparing two groups.\n""" \
               f"""You may use *plt.savefig("file_name.ext")* after the code that creates a figure to save the figure. *ext* can be png, svg, etc ..."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"@widgets.interact(metric=TIME_PLOTS)\n" \
               f"def mean_time_plots(metric):\n" \
               f"    ax = sns.lineplot(x='time (min)', y=metric, hue='experimental group', data=all_data, errorbar='{parameters.get('errorbar', RCBBatchProcess.ERRORBAR_BM)}', err_style='{parameters.get('err_style_value', RCBBatchProcess.ERR_STYLE_VALUE_BM)}', lw={RCBBatchProcess.LINE_WIDTH_BM}, palette=a_palette)\n" \
               f"    handles, _ = ax.get_legend_handles_labels()\n" \
               f"    ax.legend(handles=handles, labels={parameters.get('names')}, frameon=False)\n" \
               f"    sns.despine()\n" \
               f"    return"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb

    def _save_notebook_singleplots(self) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""## Plot individual curves for each of the experiments.\n""" \
               f"""You may use *plt.savefig("file_name.ext")* after the code that creates a figure to save the figure. *ext* can be png, svg, etc ..."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        for a_measurement in self.measurement_list:
            text = f"""{a_measurement} *vs*. time (min):"""
            nb['cells'].append(nbf.v4.new_markdown_cell(text))

            code = f"ax = sns.lineplot(x='time (min)', y='{a_measurement}', hue='experimental group', data=all_data, estimator=None, units='experiment index', lw={RCBBatchProcess.LINE_WIDTH_BM}, legend='full', palette=a_palette)\n" \
                   f"handles, labels = ax.get_legend_handles_labels()\n" \
                   f"ax.legend(handles=handles, labels=labels, frameon=False)\n" \
                   f"sns.despine()"
            nb['cells'].append(nbf.v4.new_code_cell(code))

            for a_group in self.group_labels:
                text = f"""{a_measurement} *vs*. time (min) for group '{a_group}':\n""" \
                       f"""(color-code indicates experiment index within the group)"""
                nb['cells'].append(nbf.v4.new_markdown_cell(text))

                code = f"thedata = all_data[all_data['experimental group'] == '{a_group}']\n" \
                       f"n_exp: int = len(numpy.unique(thedata['experiment index']).astype(int))\n" \
                       f"ax = sns.lineplot(x='time (min)', y='{a_measurement}', hue='experiment index', data=thedata, estimator=None, units='experiment index', lw={RCBBatchProcess.LINE_WIDTH_BM}, palette=sns.color_palette('husl', n_exp))\n" \
                       f"ax.legend([str(x) for x in range(n_exp)], frameon=False, title='{a_group}')\n" \
                       f"sns.despine()"
                nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb

    def _save_widget_notebook_singleplots(self) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""## Plot individual curves for each of the experiments.\n""" \
               f"""You may use *plt.savefig("file_name.ext")* after the code that creates a figure to save the figure. *ext* can be png, svg, etc ..."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        code = f"@widgets.interact(metric=TIME_PLOTS)\n" \
               f"def individual_time_plots(metric):\n" \
               f"    ax = sns.lineplot(x='time (min)', y=metric, hue='experimental group', data=all_data, estimator=None, units='experiment index', lw={RCBBatchProcess.LINE_WIDTH_BM}, legend='full', palette=a_palette)\n" \
               f"    handles, labels = ax.get_legend_handles_labels()\n" \
               f"    ax.legend(handles=handles[0:], labels=labels[0:], frameon=False)\n" \
               f"    sns.despine()\n" \
               f"    \n" \
               f"    for a_group in {self.group_labels}:\n" \
               f"        plt.figure()\n" \
               f"        thedata = all_data[all_data['experimental group'] == a_group]\n" \
               f"        n_exp: int = len(numpy.unique(thedata['experiment index']).astype(int))\n" \
               f"        ax = sns.lineplot(x='time (min)', y=metric, hue='experiment index', data=thedata, estimator=None, units='experiment index', lw={RCBBatchProcess.LINE_WIDTH_BM}, palette=sns.color_palette('husl', n_exp))\n" \
               f"        ax.legend([str(x) for x in range(n_exp)], frameon=False, title=a_group)\n" \
               f"        sns.despine()\n" \
               f"    \n" \
               f"    return"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb

    def _save_notebook_boxplots(self) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""## Box plots for summary metrics.\n""" \
               f"""You may use *plt.savefig("file_name.ext")* after the code that creates a figure to save the figure. *ext* can be png, svg, etc ..."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        # The rate constant is the same for all the time points of a given image, so we only take the first
        # value per image.
        # sort=False is important to preserve the order of the experimental groups, so that labels in the
        # plots below are correct.
        code = f"thedata = all_data.groupby(['experimental group', 'experiment index'], as_index=False, sort=False).first()"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        for a_measurement in self.box_plot_measurements:
            text = f"""{a_measurement}:"""
            nb['cells'].append(nbf.v4.new_markdown_cell(text))

            if self.plot_style_value == 'box':
                # whis is the proportion of the IQR past the low and high quartiles to extend the plot whiskers.
                # Points outside this range will be identified as outliers (and error bars will ignore them).
                code = f"ax = sns.boxplot(x='experimental group', y='{a_measurement}', data=thedata, hue='experimental group', dodge=False, whis=100000, palette=a_palette)\n"
            elif self.plot_style_value == 'violin':
                code = f"ax = sns.violinplot(x='experimental group', y='{a_measurement}', data=thedata, hue='experimental group', dodge=False)\n"

            code += f"handles, _ = ax.get_legend_handles_labels()\n" \
                    f"ax.legend(handles=handles, frameon=False)\n" \
                    f"ax = sns.stripplot(x='experimental group', y='{a_measurement}', data=thedata, color='k', alpha=0.75, size=6, dodge=False, jitter=0.05)\n" \
                    f"sns.despine()"
            nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb

    def _save_widget_notebook_boxplots(self) -> NotebookNode:
        nb: NotebookNode = nbf.v4.new_notebook()
        nb['cells'] = []

        text = f"""## Box plots for summary metrics.\n""" \
               f"""You may use *plt.savefig("file_name.ext")* after the code that creates a figure to save the figure. *ext* can be png, svg, etc ...\n""" \
               f"""Also note that, if you want to look at the *initial* value of any time features, you can replace *BOX_PLOTS* with *TIME_PLOTS* below."""
        nb['cells'].append(nbf.v4.new_markdown_cell(text))

        # The rate constant is the same for all the time points of a given image, so we only take the first
        # value per image.
        # sort=False is important to preserve the order of the experimental groups, so that labels in the
        # plots below are correct.
        code = f"thedata = all_data.groupby(['experimental group', 'experiment index'], as_index=False, sort=False).first()"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        code = f"@widgets.interact(metric=BOX_PLOTS)\n" \
               f"def box_plots(metric):\n"
        if self.plot_style_value == 'box':
            code += f"    ax = sns.boxplot(x='experimental group', y=metric, data=thedata, hue='experimental group', dodge=False, whis=100000, palette=a_palette)\n"
        elif self.plot_style_value == 'violin':
            code += f"    ax = sns.violinplot(x='experimental group', y=metric, data=thedata, hue='experimental group', dodge=False)\n"

        code += f"    handles, _ = ax.get_legend_handles_labels()\n" \
                f"    ax.legend(handles=handles, frameon=False)\n" \
                f"    ax = sns.stripplot(x='experimental group', y=metric, data=thedata, color='k', alpha=0.75, size=6, dodge=False, jitter=0.05)\n" \
                f"    sns.despine()\n" \
                f"    return"
        nb['cells'].append(nbf.v4.new_code_cell(code))

        return nb
