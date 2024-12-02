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

from functools import partial
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets

from pyjamas.rutils import RUtils


class BatchMeasureDialog(object):
    dirs: List[str] = None
    analyze_flag: bool = None
    analysis_filename_appendix: str = None
    analysis_extension: str = None
    save_results: bool = None  # Save the analysis script in each folder in which an analysis flag is saved.
    script_filename_appendix: str = None
    results_folder: str = None
    intensity_flag: bool = None  # Run intensity section of the analysis/plots?
    image_extension: str = None
    normalize_intensity_flag: int = None
    normalization_modes: List[str] = ['none', 'photobleaching (divide by image mean - the photobleaching normalization factor is 1 for the first image in the sequence)',
                                      'image background and photobleaching (subtract image mode and divide by image mean - the photobleaching normalization factor is 1 for the first image in the sequence)',
                                      'sample photobleaching (autodetect sample and divide by sample mean - the photobleaching normalization factor is 1 for the first image in the sequence)',
                                      'sample background and photobleaching (autodetect sample, subtract sample mode and divide by sample mean - the photobleaching normalization factor is 1 for the first image in the sequence)',
                                      'file (load additional image, subtract image mode and divide by image mean - the photobleaching normalization factor is 1 for the first image in the sequence)']
    t_res: float = None  # Time resolution in seconds.
    xy_res: float = None  # Spatial resolution in microns.
    index_time_zero: int = None  # Number of time points before treatment (e.g. number of images before wounding).
    max_index_time_zero = 30
    plot_flag: bool = None  # Generate and display plots.
    group_labels: List[str] = None
    err_styles: List[str] = ['band', 'bars']
    err_style_value: str = None
    plot_style_value: str = None
    plot_styles: List[str] = ['box', 'violin']
    brush_sz: int = None
    max_brush_sz: int = 15
    compile_data_flag: bool = None  # Read all data and compile into DataFrames.
    line_width: int = None
    max_line_width: int = 15

    def __init__(self):
        super().__init__()

    def setupUi(self, Dialog, parameters: Optional[dict] = None):
        from pyjamas.rcallbacks.rcbbatchprocess import RCBBatchProcess

        if (parameters is None or parameters is False) and BatchMeasureDialog.dirs is None:
            parameters = RCBBatchProcess._default_batchmeasure_parameters()

        if parameters is not None and parameters is not False:
            BatchMeasureDialog.dirs = parameters.get('folders')
            BatchMeasureDialog.analyze_flag = parameters.get('analyze_flag')
            BatchMeasureDialog.analysis_filename_appendix = parameters.get('analysis_filename_appendix')
            BatchMeasureDialog.analysis_extension = parameters.get('analysis_extension')
            BatchMeasureDialog.save_results = parameters.get('save_results')  # Save the analysis script in each folder in which an analysis flag is saved.
            BatchMeasureDialog.script_filename_appendix = parameters.get('script_filename_appendix')
            BatchMeasureDialog.results_folder = parameters.get('results_folder')
            BatchMeasureDialog.intensity_flag = parameters.get('intensity_flag')  # Run intensity section of the analysis/plots?
            BatchMeasureDialog.image_extension = parameters.get('image_extension')
            BatchMeasureDialog.normalize_intensity_flag = parameters.get('normalize_intensity_flag')
            BatchMeasureDialog.t_res = parameters.get('t_res')  # Time resolution in seconds.
            BatchMeasureDialog.xy_res = parameters.get('xy_res')  # Spatial resolution in microns.
            BatchMeasureDialog.index_time_zero = parameters.get('index_time_zero')  # Number of time points before treatment (e.g. number of images before wounding).
            BatchMeasureDialog.plot_flag = parameters.get('plot_flag')  # Generate and display plots.
            BatchMeasureDialog.group_labels = parameters.get('names')
            BatchMeasureDialog.err_style_value = parameters.get('err_style_value')
            BatchMeasureDialog.plot_style_value = parameters.get('plot_style_value')
            BatchMeasureDialog.brush_sz = parameters.get('brush_sz')

        Dialog.setObjectName("Dialog")
        Dialog.resize(704, 530)
        self.buttonsOkCancel = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonsOkCancel.setGeometry(QtCore.QRect(310, 490, 341, 32))
        self.buttonsOkCancel.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonsOkCancel.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonsOkCancel.setObjectName("buttonsOkCancel")

        self.GroupTabBox = QtWidgets.QTabWidget(Dialog)
        self.GroupTabBox.setGeometry(QtCore.QRect(20, 10, 661, 91))
        self.GroupTabBox.setObjectName("groupTabBox")

        self.dir_groupboxes = []
        self.dir_labels = []
        self.dir_folderlines = []
        self.dir_pbs = []
        self.dir_namelines = []
        self.dir_namelabels = []

        for dir_index, dir_path in enumerate(self.dirs):
            self.dir_groupboxes.append(QtWidgets.QGroupBox())
            self.dir_groupboxes[dir_index].setGeometry(QtCore.QRect(1, 1, 659, 89))
            self.dir_groupboxes[dir_index].setObjectName(f"dir_groupbox_{dir_index}")
            self.dir_labels.append(QtWidgets.QLabel(self.dir_groupboxes[dir_index]))
            self.dir_labels[dir_index].setGeometry(QtCore.QRect(10, 10, 131, 31))
            self.dir_labels[dir_index].setObjectName(f"dir_label_{dir_index}")
            self.dir_folderlines.append(QtWidgets.QLineEdit(self.dir_groupboxes[dir_index]))
            self.dir_folderlines[dir_index].setGeometry(QtCore.QRect(165, 15, 271, 21))
            self.dir_folderlines[dir_index].setObjectName(f"leFolder_{dir_index}")
            self.dir_pbs.append(QtWidgets.QPushButton(self.dir_groupboxes[dir_index]))
            self.dir_pbs[dir_index].setGeometry(QtCore.QRect(432, 11, 51, 32))
            self.dir_pbs[dir_index].setObjectName(f"pbFolder_{dir_index}")
            self.dir_pbs[dir_index].clicked.connect(partial(self._open_folder_dialog, dir_index))
            self.dir_namelines.append(QtWidgets.QLineEdit(self.dir_groupboxes[dir_index]))
            self.dir_namelines[dir_index].setGeometry(QtCore.QRect(550, 15, 101, 21))
            self.dir_namelines[dir_index].setObjectName(f"leName_{dir_index}")
            self.dir_namelabels.append(QtWidgets.QLabel(self.dir_groupboxes[dir_index]))
            self.dir_namelabels[dir_index].setGeometry(QtCore.QRect(499, 10, 131, 31))
            self.dir_namelabels[dir_index].setObjectName(f"name_label_{dir_index}")
            self.GroupTabBox.addTab(self.dir_groupboxes[dir_index], f"{dir_index+1}")
            self.dir_folderlines[dir_index].raise_()
            self.dir_namelines[dir_index].raise_()

        self.addTabPb = QtWidgets.QPushButton(Dialog)
        self.addTabPb.setGeometry(QtCore.QRect(20, 105, 100, 32))
        self.addTabPb.setObjectName("addtab_pb")
        self.addTabPb.clicked.connect(partial(self._add_tab, Dialog))
        self.removeTabPb = QtWidgets.QPushButton(Dialog)
        self.removeTabPb.setGeometry(QtCore.QRect(130, 105, 100, 32))
        self.removeTabPb.setObjectName("removetab_pb")
        self.removeTabPb.clicked.connect(partial(self._remove_tab, Dialog))

        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 130, 531, 351))
        self.groupBox_2.setObjectName("groupBox_2")
        self.cbRunAnalysis = QtWidgets.QCheckBox(self.groupBox_2)
        self.cbRunAnalysis.setGeometry(QtCore.QRect(10, 237, 101, 20))
        self.cbRunAnalysis.setObjectName("cbRunAnalysis")
        self.cbAnalyzeIntensities = QtWidgets.QCheckBox(self.groupBox_2)
        self.cbAnalyzeIntensities.setGeometry(QtCore.QRect(30, 84, 181, 20))
        self.cbAnalyzeIntensities.setObjectName("cbAnalyzeIntensities")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(30, 157, 171, 31))
        self.label_5.setObjectName("label_5")
        self.leAnalysisFileSuffix = QtWidgets.QLineEdit(self.groupBox_2)
        self.leAnalysisFileSuffix.setGeometry(QtCore.QRect(210, 162, 271, 21))
        self.leAnalysisFileSuffix.setObjectName("leAnalysisFileSuffix")
        self.leAnalysisFileExtension = QtWidgets.QLineEdit(self.groupBox_2)
        self.leAnalysisFileExtension.setGeometry(QtCore.QRect(210, 187, 271, 21))
        self.leAnalysisFileExtension.setObjectName("leAnalysisFileExtension")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(30, 182, 171, 31))
        self.label_6.setObjectName("label_6")
        self.leScriptFileSuffix = QtWidgets.QLineEdit(self.groupBox_2)
        self.leScriptFileSuffix.setGeometry(QtCore.QRect(200, 302, 271, 21))
        self.leScriptFileSuffix.setObjectName("leScriptFileSuffix")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(50, 272, 171, 31))
        self.label_7.setObjectName("label_7")
        self.cbSaveResults = QtWidgets.QCheckBox(self.groupBox_2)
        self.cbSaveResults.setGeometry(QtCore.QRect(10, 259, 181, 20))
        self.cbSaveResults.setObjectName("cbSaveResults")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(50, 297, 171, 31))
        self.label_8.setObjectName("label_8")
        self.leFolderResults = QtWidgets.QLineEdit(self.groupBox_2)
        self.leFolderResults.setGeometry(QtCore.QRect(200, 277, 271, 21))
        self.leFolderResults.setObjectName("leFolderResults")
        self.pbFolderResults = QtWidgets.QPushButton(Dialog)
        self.pbFolderResults.setGeometry(QtCore.QRect(492, 402, 51, 32))
        self.pbFolderResults.setObjectName("pbFolderResults")
        self.pbFolderResults.clicked.connect(self._open_results_folder_dialog)
        self.leImageExtension = QtWidgets.QLineEdit(self.groupBox_2)
        self.leImageExtension.setGeometry(QtCore.QRect(170, 109, 31, 21))
        self.leImageExtension.setObjectName("leImageExtension")
        self.label_13 = QtWidgets.QLabel(self.groupBox_2)
        self.label_13.setGeometry(QtCore.QRect(60, 104, 171, 31))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.groupBox_2)
        self.label_14.setGeometry(QtCore.QRect(30, 132, 171, 31))
        self.label_14.setObjectName("label_14")
        self.cbNormalization = QtWidgets.QComboBox(self.groupBox_2)
        self.cbNormalization.setGeometry(QtCore.QRect(205, 137, 281, 26))
        self.cbNormalization.setObjectName("cbNormalization")
        for a_mode in BatchMeasureDialog.normalization_modes:
            self.cbNormalization.addItem(a_mode)
        self.label_15 = QtWidgets.QLabel(self.groupBox_2)
        self.label_15.setGeometry(QtCore.QRect(30, 23, 171, 31))
        self.label_15.setObjectName("label_15")
        self.leTimeResolution = QtWidgets.QLineEdit(self.groupBox_2)
        self.leTimeResolution.setGeometry(QtCore.QRect(190, 28, 51, 21))
        self.leTimeResolution.setObjectName("leTimeResolution")
        self.leXYResolution = QtWidgets.QLineEdit(self.groupBox_2)
        self.leXYResolution.setGeometry(QtCore.QRect(190, 57, 51, 21))
        self.leXYResolution.setObjectName("leXYResolution")
        self.label_16 = QtWidgets.QLabel(self.groupBox_2)
        self.label_16.setGeometry(QtCore.QRect(30, 52, 141, 31))
        self.label_16.setObjectName("label_16")
        self.cbPlotResults = QtWidgets.QCheckBox(self.groupBox_2)
        self.cbPlotResults.setGeometry(QtCore.QRect(10, 324, 181, 20))
        self.cbPlotResults.setObjectName("cbPlotResults")
        self.label_17 = QtWidgets.QLabel(self.groupBox_2)
        self.label_17.setGeometry(QtCore.QRect(280, 52, 121, 31))
        self.label_17.setObjectName("label_17")
        self.cbBrushSz = QtWidgets.QComboBox(self.groupBox_2)
        self.cbBrushSz.setGeometry(QtCore.QRect(400, 56, 104, 26))
        self.cbBrushSz.setObjectName("cbBrushSz")
        for i in range(BatchMeasureDialog.max_brush_sz):
            self.cbBrushSz.addItem(str(i+1))
        self.cbErrorStyle = QtWidgets.QComboBox(self.groupBox_2)
        self.cbErrorStyle.setGeometry(QtCore.QRect(120, 212, 104, 26))
        self.cbErrorStyle.setObjectName("cbErrorStyle")
        for a_style in BatchMeasureDialog.err_styles:
            self.cbErrorStyle.addItem(a_style)
        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        self.label_18.setGeometry(QtCore.QRect(55, 207, 71, 31))
        self.label_18.setObjectName("label_18")
        self.label_40 = QtWidgets.QLabel(self.groupBox_2)
        self.label_40.setGeometry(QtCore.QRect(280, 23, 171, 31))
        self.label_40.setObjectName("label_40")
        self.cbIndexZero = QtWidgets.QComboBox(self.groupBox_2)
        self.cbIndexZero.setGeometry(QtCore.QRect(400, 23, 104, 26))
        self.cbIndexZero.setObjectName("cbIndexZero")
        for i in range(self.max_index_time_zero+1):
            self.cbIndexZero.addItem(str(i))
        self.cbPlotStyle = QtWidgets.QComboBox(self.groupBox_2)
        self.cbPlotStyle.setGeometry(QtCore.QRect(315, 212, 104, 26))
        self.cbPlotStyle.setObjectName("cbPlotStyle")
        for a_style in BatchMeasureDialog.plot_styles:
            self.cbPlotStyle.addItem(a_style)
        self.label_41 = QtWidgets.QLabel(self.groupBox_2)
        self.label_41.setGeometry(QtCore.QRect(250, 207, 71, 31))
        self.label_41.setObjectName("label_41")
        self.label_13.raise_()
        self.cbRunAnalysis.raise_()
        self.cbAnalyzeIntensities.raise_()
        self.label_5.raise_()
        self.leAnalysisFileSuffix.raise_()
        self.leAnalysisFileExtension.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.cbSaveResults.raise_()
        self.label_8.raise_()
        self.leScriptFileSuffix.raise_()
        self.leFolderResults.raise_()
        self.leImageExtension.raise_()
        self.label_14.raise_()
        self.cbNormalization.raise_()
        self.label_15.raise_()
        self.leTimeResolution.raise_()
        self.leXYResolution.raise_()
        self.label_16.raise_()
        self.cbPlotResults.raise_()
        self.label_17.raise_()
        self.cbBrushSz.raise_()
        self.cbErrorStyle.raise_()
        self.label_18.raise_()
        self.label_40.raise_()
        self.cbIndexZero.raise_()
        self.cbPlotStyle.raise_()
        self.label_41.raise_()
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(560, 130, 120, 351))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_19 = QtWidgets.QLabel(self.groupBox_3)
        self.label_19.setGeometry(QtCore.QRect(10, 20, 51, 31))
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox_3)
        self.label_20.setGeometry(QtCore.QRect(10, 51, 51, 31))
        self.label_20.setObjectName("label_20")
        self.label_21 = QtWidgets.QLabel(self.groupBox_3)
        self.label_21.setGeometry(QtCore.QRect(10, 65, 81, 31))
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.groupBox_3)
        self.label_22.setGeometry(QtCore.QRect(10, 81, 91, 31))
        self.label_22.setObjectName("label_22")
        self.label_23 = QtWidgets.QLabel(self.groupBox_3)
        self.label_23.setGeometry(QtCore.QRect(10, 123, 101, 31))
        self.label_23.setObjectName("label_23")
        self.label_24 = QtWidgets.QLabel(self.groupBox_3)
        self.label_24.setGeometry(QtCore.QRect(10, 109, 51, 31))
        self.label_24.setObjectName("label_24")
        self.label_25 = QtWidgets.QLabel(self.groupBox_3)
        self.label_25.setGeometry(QtCore.QRect(10, 139, 101, 31))
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.groupBox_3)
        self.label_26.setGeometry(QtCore.QRect(10, 96, 91, 31))
        self.label_26.setObjectName("label_26")
        self.label_27 = QtWidgets.QLabel(self.groupBox_3)
        self.label_27.setGeometry(QtCore.QRect(10, 153, 91, 31))
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.groupBox_3)
        self.label_28.setGeometry(QtCore.QRect(10, 166, 51, 31))
        self.label_28.setObjectName("label_28")
        self.label_29 = QtWidgets.QLabel(self.groupBox_3)
        self.label_29.setGeometry(QtCore.QRect(10, 180, 101, 31))
        self.label_29.setObjectName("label_29")
        self.label_30 = QtWidgets.QLabel(self.groupBox_3)
        self.label_30.setGeometry(QtCore.QRect(10, 196, 111, 31))
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.groupBox_3)
        self.label_31.setGeometry(QtCore.QRect(10, 212, 91, 31))
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.groupBox_3)
        self.label_32.setGeometry(QtCore.QRect(10, 225, 51, 31))
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(self.groupBox_3)
        self.label_33.setGeometry(QtCore.QRect(10, 255, 111, 31))
        self.label_33.setObjectName("label_33")
        self.label_34 = QtWidgets.QLabel(self.groupBox_3)
        self.label_34.setGeometry(QtCore.QRect(10, 239, 111, 31))
        self.label_34.setObjectName("label_34")
        self.label_35 = QtWidgets.QLabel(self.groupBox_3)
        self.label_35.setGeometry(QtCore.QRect(25, 293, 91, 31))
        self.label_35.setObjectName("label_35")
        self.label_36 = QtWidgets.QLabel(self.groupBox_3)
        self.label_36.setGeometry(QtCore.QRect(10, 266, 91, 31))
        self.label_36.setObjectName("label_36")
        self.label_37 = QtWidgets.QLabel(self.groupBox_3)
        self.label_37.setGeometry(QtCore.QRect(25, 309, 91, 31))
        self.label_37.setObjectName("label_37")
        self.label_38 = QtWidgets.QLabel(self.groupBox_3)
        self.label_38.setGeometry(QtCore.QRect(10, 279, 51, 31))
        self.label_38.setObjectName("label_38")
        self.label_39 = QtWidgets.QLabel(self.groupBox_3)
        self.label_39.setGeometry(QtCore.QRect(10, 35, 51, 31))
        self.label_39.setObjectName("label_39")
        self.buttonsOkCancel.raise_()
        self.groupBox_2.raise_()
        self.groupBox_3.raise_()
        self.pbFolderResults.raise_()

        self.retranslateUi(Dialog, overwrite_inputs=True)
        self.buttonsOkCancel.accepted.connect(Dialog.accept)
        self.buttonsOkCancel.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog, overwrite_inputs: bool = True):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Measure"))

        for d, this_dir in enumerate(self.dirs):
            self.dir_labels[d].setText(_translate("Dialog", f"experiment folder {d+1}"))
            self.dir_pbs[d].setText(_translate("Dialog", "..."))
            self.dir_namelabels[d].setText(_translate("Dialog", "name"))
            if overwrite_inputs:
                self.dir_folderlines[d].setText(BatchMeasureDialog.dirs[d])
                self.dir_namelines[d].setText(_translate("Dialog", BatchMeasureDialog.group_labels[d]))

        self.addTabPb.setText(_translate("Dialog", "Add group"))
        self.removeTabPb.setText(_translate("Dialog", "Remove group"))

        self.groupBox_2.setTitle(_translate("Dialog", "Parameters"))
        self.cbRunAnalysis.setText(_translate("Dialog", "run analysis"))
        self.cbAnalyzeIntensities.setText(_translate("Dialog", "analyze image intensities"))
        self.label_5.setText(_translate("Dialog", "analysis file name suffix"))
        self.label_6.setText(_translate("Dialog", "analysis file name extension"))
        self.label_7.setText(_translate("Dialog", "results folder"))
        self.pbFolderResults.setText(_translate("Dialog", "..."))
        self.cbSaveResults.setText(_translate("Dialog", "save analysis results"))
        self.label_8.setText(_translate("Dialog", "notebook suffix"))
        self.label_13.setText(_translate("Dialog", "image extension"))
        self.label_14.setText(_translate("Dialog", "intensity normalization"))
        self.label_15.setText(_translate("Dialog", "time resolution (seconds)"))
        self.label_16.setText(_translate("Dialog", "xy resolution (microns)"))
        self.cbPlotResults.setText(_translate("Dialog", "plot results"))
        self.label_17.setText(_translate("Dialog", "brush size (pixels)"))
        self.label_18.setText(_translate("Dialog", "error style"))
        self.label_40.setText(_translate("Dialog", "images before \ntreatment"))
        self.label_41.setText(_translate("Dialog", "plot style"))
        self.groupBox_3.setTitle(_translate("Dialog", "Sample folder"))
        self.label_19.setText(_translate("Dialog", "control"))
        self.label_20.setText(_translate("Dialog", "|-n1"))
        self.label_21.setText(_translate("Dialog", "|   |-image.tif"))
        self.label_22.setText(_translate("Dialog", "|   |-image.pjs"))
        self.label_23.setText(_translate("Dialog", "|   |-myimage.tif"))
        self.label_24.setText(_translate("Dialog", "|-n2"))
        self.label_25.setText(_translate("Dialog", "|   |-myimage.pjs"))
        self.label_26.setText(_translate("Dialog", "| "))
        self.label_27.setText(_translate("Dialog", "| "))
        self.label_28.setText(_translate("Dialog", "|-n3"))
        self.label_29.setText(_translate("Dialog", "|   |-201914.tif"))
        self.label_30.setText(_translate("Dialog", "|   |-201914.pjs"))
        self.label_31.setText(_translate("Dialog", "| "))
        self.label_32.setText(_translate("Dialog", "|-a3592"))
        self.label_33.setText(_translate("Dialog", "|   |-a3592.pjs"))
        self.label_34.setText(_translate("Dialog", "|   |-a3592.tif"))
        self.label_35.setText(_translate("Dialog", "|-a12_3.tif"))
        self.label_36.setText(_translate("Dialog", "| "))
        self.label_37.setText(_translate("Dialog", "|-a12_3.pjs"))
        self.label_38.setText(_translate("Dialog", "|-exp36"))
        self.label_39.setText(_translate("Dialog", "|"))

        if overwrite_inputs:
            self.cbRunAnalysis.setChecked(BatchMeasureDialog.analyze_flag)
            self.cbAnalyzeIntensities.setChecked(BatchMeasureDialog.intensity_flag)
            self.leAnalysisFileSuffix.setText(_translate("Dialog", BatchMeasureDialog.analysis_filename_appendix))
            self.leAnalysisFileExtension.setText(_translate("Dialog", BatchMeasureDialog.analysis_extension))
            self.leFolderResults.setText(_translate("Dialog", BatchMeasureDialog.results_folder))
            self.cbSaveResults.setChecked(BatchMeasureDialog.save_results)
            self.leScriptFileSuffix.setText(_translate("Dialog", BatchMeasureDialog.script_filename_appendix))
            self.leImageExtension.setText(_translate("Dialog", BatchMeasureDialog.image_extension))
            self.cbNormalization.setCurrentIndex(int(BatchMeasureDialog.normalize_intensity_flag))
            self.leTimeResolution.setText(str(BatchMeasureDialog.t_res))
            self.leXYResolution.setText(str(round(BatchMeasureDialog.xy_res*1000)/1000.))
            self.cbPlotResults.setChecked(BatchMeasureDialog.plot_flag)
            self.cbBrushSz.setCurrentIndex(BatchMeasureDialog.brush_sz-1)
            self.cbErrorStyle.setCurrentIndex(BatchMeasureDialog.err_styles.index(BatchMeasureDialog.err_style_value))
            self.cbIndexZero.setCurrentIndex(BatchMeasureDialog.index_time_zero)
            self.cbPlotStyle.setCurrentIndex(BatchMeasureDialog.plot_styles.index(BatchMeasureDialog.plot_style_value))

    def parameters(self) -> dict:

        BatchMeasureDialog.dirs: List[str] = [this_folder.text() for this_folder in self.dir_folderlines]
        BatchMeasureDialog.analyze_flag: bool = self.cbRunAnalysis.isChecked()
        BatchMeasureDialog.analysis_filename_appendix: str = self.leAnalysisFileSuffix.text()
        BatchMeasureDialog.analysis_extension: str = self.leAnalysisFileExtension.text()
        BatchMeasureDialog.results_folder: str = self.leFolderResults.text()
        BatchMeasureDialog.save_results: bool = self.cbSaveResults.isChecked()  # Save the analysis script in each folder in which an analysis flag is saved.
        BatchMeasureDialog.script_filename_appendix: str = self.leScriptFileSuffix.text()
        BatchMeasureDialog.intensity_flag: bool = self.cbAnalyzeIntensities.isChecked()  # Run intensity section of the analysis/plots?
        BatchMeasureDialog.image_extension: str = self.leImageExtension.text()
        BatchMeasureDialog.normalize_intensity_flag = self.cbNormalization.currentIndex()
        BatchMeasureDialog.t_res: float = float(self.leTimeResolution.text())  # Time resolution in seconds.
        BatchMeasureDialog.xy_res: float = float(self.leXYResolution.text())  # Spatial resolution in microns.
        BatchMeasureDialog.index_time_zero: int = self.cbIndexZero.currentIndex() # Number of time points before treatment (e.g. number of images before wounding).
        BatchMeasureDialog.plot_flag: bool = self.cbPlotResults.isChecked()  # Generate and display plots.
        BatchMeasureDialog.group_labels: List[str] = [this_name.text() for this_name in self.dir_namelines]
        BatchMeasureDialog.err_style_value: str = self.cbErrorStyle.currentText()
        BatchMeasureDialog.plot_style_value: str = self.cbPlotStyle.currentText()
        BatchMeasureDialog.brush_sz: int = int(self.cbBrushSz.currentText())

        return {
            'folders': BatchMeasureDialog.dirs,
            'analyze_flag': BatchMeasureDialog.analyze_flag,
            'analysis_filename_appendix': BatchMeasureDialog.analysis_filename_appendix,
            'analysis_extension': BatchMeasureDialog.analysis_extension,
            'save_results': BatchMeasureDialog.save_results,
            'script_filename_appendix': BatchMeasureDialog.script_filename_appendix,
            'results_folder': BatchMeasureDialog.results_folder,
            'intensity_flag': BatchMeasureDialog.intensity_flag,
            'image_extension': BatchMeasureDialog.image_extension,
            'normalize_intensity_flag': BatchMeasureDialog.normalize_intensity_flag,
            't_res': BatchMeasureDialog.t_res,
            'xy_res': BatchMeasureDialog.xy_res,
            'index_time_zero': BatchMeasureDialog.index_time_zero,
            'plot_flag': BatchMeasureDialog.plot_flag,
            'names': BatchMeasureDialog.group_labels,
            'err_style_value': BatchMeasureDialog.err_style_value,
            'plot_style_value': BatchMeasureDialog.plot_style_value,
            'brush_sz': BatchMeasureDialog.brush_sz,
        }

    def _open_folder_dialog(self, dir_num: int = 0) -> bool:
        start_folder = self.dir_folderlines[dir_num].text() if self.dir_folderlines[dir_num].text() != '' else BatchMeasureDialog.dirs[dir_num]
        folder = RUtils.open_folder_dialog(f"{BatchMeasureDialog.group_labels[dir_num]} folder", start_folder)

        if folder == '' or folder is False:
            return False

        self.dir_folderlines[dir_num].setText(folder)
        return True

    def _add_tab(self, Dialog):
        self.dirs.append("")
        self.group_labels.append(f"group {len(self.dirs)}")
        self.dir_groupboxes.append(QtWidgets.QGroupBox())
        self.dir_groupboxes[-1].setGeometry(QtCore.QRect(1, 1, 659, 89))
        self.dir_groupboxes[-1].setObjectName(f"dir_groupbox_{len(self.dir_groupboxes) - 1}")
        self.dir_labels.append(QtWidgets.QLabel(self.dir_groupboxes[-1]))
        self.dir_labels[-1].setGeometry(QtCore.QRect(10, 10, 131, 31))
        self.dir_labels[-1].setObjectName(f"dir_label_{len(self.dir_labels) - 1}")
        self.dir_folderlines.append(QtWidgets.QLineEdit(self.dir_groupboxes[-1]))
        self.dir_folderlines[-1].setGeometry(QtCore.QRect(165, 15, 271, 21))
        self.dir_folderlines[-1].setObjectName(f"leFolder_{len(self.dir_folderlines) - 1}")
        self.dir_pbs.append(QtWidgets.QPushButton(self.dir_groupboxes[-1]))
        self.dir_pbs[-1].setGeometry(QtCore.QRect(432, 11, 51, 32))
        self.dir_pbs[-1].setObjectName(f"pbFolder_{len(self.dir_pbs) - 1}")
        self.dir_pbs[-1].clicked.connect(partial(self._open_folder_dialog, len(self.dir_pbs) - 1))
        self.dir_namelines.append(QtWidgets.QLineEdit(self.dir_groupboxes[-1]))
        self.dir_namelines[-1].setGeometry(QtCore.QRect(550, 15, 101, 21))
        self.dir_namelines[-1].setObjectName(f"leName_{len(self.dir_namelines) - 1}")
        self.dir_namelabels.append(QtWidgets.QLabel(self.dir_groupboxes[-1]))
        self.dir_namelabels[-1].setGeometry(QtCore.QRect(499, 10, 131, 31))
        self.dir_namelabels[-1].setObjectName(f"name_label_{len(self.dir_namelabels) - 1}")
        self.GroupTabBox.addTab(self.dir_groupboxes[-1], f"{len(self.dir_namelabels)}")
        self.dir_folderlines[-1].raise_()
        self.dir_namelines[-1].raise_()
        self.dir_folderlines[-1].setText(self.dirs[-1])
        self.dir_namelines[-1].setText(self.group_labels[-1])
        self.retranslateUi(Dialog, overwrite_inputs=False)

        return True

    def _remove_tab(self, Dialog):
        if len(self.dirs) == 1:
            return False

        index = self.GroupTabBox.currentIndex()
        self.dirs.pop(index)
        self.group_labels.pop(index)
        self.dir_groupboxes.pop(index)
        self.dir_labels.pop(index)
        self.dir_folderlines.pop(index)
        self.dir_pbs.pop(index)
        self.dir_namelines.pop(index)
        self.dir_namelabels.pop(index)
        self.GroupTabBox.removeTab(index)
        for this_tab in range(index, len(self.dirs)):  # rename and reconnect all tabs to the right
            self.GroupTabBox.setTabText(this_tab, f"{this_tab+1}")
            self.dir_pbs[this_tab].clicked.disconnect()
            self.dir_pbs[this_tab].clicked.connect(partial(self._open_folder_dialog, this_tab))
        self.retranslateUi(Dialog, overwrite_inputs=False)

        return True

    def _open_results_folder_dialog(self) -> bool:
        start_folder = self.dir_folderlines[0].text() if self.dir_folderlines[
                                                             0].text() != '' else BatchMeasureDialog.results_folder
        folder = RUtils.open_folder_dialog(f"Results folder", start_folder)

        if folder == '' or folder is False:
            return False

        self.leFolderResults.setText(folder)

        return True