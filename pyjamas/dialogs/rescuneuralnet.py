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
from typing import Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from pyjamas.dialogs.classifierdialogABC import ClassifierDialogABC
from pyjamas.rimage.rimml.rimrescunet import ReSCUNet
from pyjamas.rutils import RUtils


class ReSCUNeuralNetDialog(ClassifierDialogABC):
    epochs: int = ReSCUNet.EPOCHS
    learning_rate: float = ReSCUNet.LEARNING_RATE
    mini_batch_size: int = ReSCUNet.BATCH_SIZE
    concatenation_level: int = ReSCUNet.CONCATENATION_LEVEL
    erosion_width: int = ReSCUNet.EROSION_WIDTH
    generate_notebook: bool = True
    notebook_path: str = ''
    step_sz: Tuple[int, int] = ReSCUNet.STEP_SIZE

    def __init__(self):
        super().__init__()

    def setupUi(self, NNet, parameters: Optional[dict] = None):
        if parameters is None or parameters is False:
            parameters = {
                'positive_training_folder': ReSCUNeuralNetDialog.positive_training_folder,
                'train_image_size': ReSCUNeuralNetDialog.train_image_size,
                'step_sz': ReSCUNeuralNetDialog.step_sz,
                'epochs': ReSCUNeuralNetDialog.epochs,
                'learning_rate': ReSCUNeuralNetDialog.learning_rate,
                'mini_batch_size': ReSCUNeuralNetDialog.mini_batch_size,
                'concatenation_level': ReSCUNeuralNetDialog.concatenation_level,
                'generate_notebook': ReSCUNeuralNetDialog.generate_notebook,
                'erosion_width': ReSCUNeuralNetDialog.erosion_width,
                'notebook_path': ReSCUNeuralNetDialog.notebook_path,
            }

        ReSCUNeuralNetDialog.positive_training_folder = parameters.get('positive_training_folder', ReSCUNeuralNetDialog.positive_training_folder)
        ReSCUNeuralNetDialog.train_image_size = parameters.get('train_image_size', ReSCUNeuralNetDialog.train_image_size)
        ReSCUNeuralNetDialog.step_sz = parameters.get('step_sz', ReSCUNeuralNetDialog.step_sz)
        ReSCUNeuralNetDialog.epochs = parameters.get('epochs', ReSCUNeuralNetDialog.epochs)
        ReSCUNeuralNetDialog.learning_rate = parameters.get('learning_rate', ReSCUNeuralNetDialog.learning_rate)
        ReSCUNeuralNetDialog.mini_batch_size = parameters.get('mini_batch_size', ReSCUNeuralNetDialog.mini_batch_size)
        ReSCUNeuralNetDialog.generate_notebook = parameters.get('generate_notebook', ReSCUNeuralNetDialog.generate_notebook)
        ReSCUNeuralNetDialog.erosion_width = parameters.get('erosion_width', ReSCUNeuralNetDialog.erosion_width)
        ReSCUNeuralNetDialog.notebook_path = parameters.get('notebook_path', ReSCUNeuralNetDialog.notebook_path)
        ReSCUNeuralNetDialog.concatenation_level = parameters.get('concatenation_level', ReSCUNeuralNetDialog.concatenation_level)

        NNet.setObjectName("NNet")
        NNet.resize(614, 375)
        self.buttonBox = QtWidgets.QDialogButtonBox(NNet)
        self.buttonBox.setGeometry(QtCore.QRect(240, 325, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.groupBox_2 = QtWidgets.QGroupBox(NNet)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 26, 551, 66))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(31, 26, 141, 24))
        self.label.setObjectName("label")
        self.positive_training_folder_edit = QtWidgets.QLineEdit(self.groupBox_2)
        self.positive_training_folder_edit.setGeometry(QtCore.QRect(220, 30, 261, 21))
        self.positive_training_folder_edit.setObjectName("positive_training_folder_edit")
        self.positive_training_folder_edit.setText(ReSCUNeuralNetDialog.positive_training_folder)
        self.btnSavePositive = QtWidgets.QToolButton(self.groupBox_2)
        self.btnSavePositive.setGeometry(QtCore.QRect(490, 30, 26, 22))
        self.btnSavePositive.setObjectName("btnSavePositive")
        self.btnSavePositive.clicked.connect(self._open_positive_folder_dialog)
        self.groupBox_3 = QtWidgets.QGroupBox(NNet)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 101, 251, 61))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(31, 28, 141, 24))
        self.label_4.setObjectName("label_4")
        self.lnWidth = QtWidgets.QLineEdit(self.groupBox_3)
        self.lnWidth.setGeometry(QtCore.QRect(70, 30, 31, 21))
        self.lnWidth.setObjectName("lnWidth")
        self.lnWidth.setText(str(ReSCUNeuralNetDialog.train_image_size[1]))
        self.lnHeight = QtWidgets.QLineEdit(self.groupBox_3)
        self.lnHeight.setGeometry(QtCore.QRect(170, 30, 31, 21))
        self.lnHeight.setObjectName("lnHeight")
        self.lnHeight.setText(str(ReSCUNeuralNetDialog.train_image_size[0]))
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(120, 28, 141, 24))
        self.label_5.setObjectName("label_5")
        self.label_5.raise_()
        self.label_4.raise_()
        self.lnWidth.raise_()
        self.lnHeight.raise_()
        self.groupBox_5 = QtWidgets.QGroupBox(NNet)
        self.groupBox_5.setGeometry(QtCore.QRect(30, 171, 551, 150))
        self.groupBox_5.setObjectName("groupBox_5")
        self.lnEpochs = QtWidgets.QLineEdit(self.groupBox_5)
        self.lnEpochs.setGeometry(QtCore.QRect(375, 27, 41, 21))
        self.lnEpochs.setObjectName("lnEpochs")
        self.lnEpochs.setText(str(ReSCUNeuralNetDialog.epochs))
        self.label_10 = QtWidgets.QLabel(self.groupBox_5)
        self.label_10.setGeometry(QtCore.QRect(325, 31, 45, 16))
        self.label_10.setObjectName("label_10")
        self.lnEta = QtWidgets.QLineEdit(self.groupBox_5)
        self.lnEta.setGeometry(QtCore.QRect(96, 27, 46, 21))
        self.lnEta.setObjectName("lnEta")
        self.lnEta.setText(str(ReSCUNeuralNetDialog.learning_rate))
        self.label_11 = QtWidgets.QLabel(self.groupBox_5)
        self.label_11.setGeometry(QtCore.QRect(10, 31, 141, 16))
        self.label_11.setObjectName("label_11")
        self.lnBatchSz = QtWidgets.QLineEdit(self.groupBox_5)
        self.lnBatchSz.setGeometry(QtCore.QRect(250, 27, 36, 21))
        self.lnBatchSz.setObjectName("lnBatchSz")
        self.lnBatchSz.setText(str(ReSCUNeuralNetDialog.mini_batch_size))
        self.label_14 = QtWidgets.QLabel(self.groupBox_5)
        self.label_14.setGeometry(QtCore.QRect(180, 31, 95, 16))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox_5)
        self.label_15.setGeometry(QtCore.QRect(220, 75, 141, 16))
        self.label_15.setObjectName("label_15")
        self.lnErosionWidth = QtWidgets.QLineEdit(self.groupBox_5)
        self.lnErosionWidth.setGeometry(QtCore.QRect(310, 71, 46, 21))
        self.lnErosionWidth.setObjectName("lnErosionWidth")
        self.lnErosionWidth.setText(str(ReSCUNeuralNetDialog.erosion_width))
        self.label_16 = QtWidgets.QLabel(self.groupBox_5)
        self.label_16.setGeometry(QtCore.QRect(10, 75, 141, 16))
        self.label_16.setObjectName("label_16")
        self.lnConcatenationLevel = QtWidgets.QLineEdit(self.groupBox_5)
        self.lnConcatenationLevel.setGeometry(QtCore.QRect(150, 71, 46, 21))
        self.lnConcatenationLevel.setObjectName("lnConcatenationLevel")
        self.lnConcatenationLevel.setText(str(ReSCUNeuralNetDialog.concatenation_level))

        self.cbGenerateNotebook = QtWidgets.QCheckBox(self.groupBox_5)
        self.cbGenerateNotebook.setGeometry(QtCore.QRect(10, 115, 200, 16))
        self.cbGenerateNotebook.setObjectName("cbGenerateNotebook")
        self.cbGenerateNotebook.setChecked(ReSCUNeuralNetDialog.generate_notebook)
        self.notebook_path_edit = QtWidgets.QLineEdit(self.groupBox_5)
        self.notebook_path_edit.setGeometry(QtCore.QRect(220, 110, 261, 21))
        self.notebook_path_edit.setObjectName("notebook_path_edit")
        self.notebook_path_edit.setText(ReSCUNeuralNetDialog.notebook_path)
        self.btnSaveNotebook = QtWidgets.QToolButton(self.groupBox_5)
        self.btnSaveNotebook.setGeometry(QtCore.QRect(490, 110, 26, 22))
        self.btnSaveNotebook.setObjectName("btnSaveNotebook")
        self.btnSaveNotebook.clicked.connect(self._open_notebook_path_dialog)

        self.label_10.raise_()
        self.lnEpochs.raise_()
        self.label_11.raise_()
        self.lnEta.raise_()
        self.label_14.raise_()
        self.lnBatchSz.raise_()
        self.groupBox_6 = QtWidgets.QGroupBox(NNet)
        self.groupBox_6.setGeometry(QtCore.QRect(300, 100, 281, 61))
        self.groupBox_6.setObjectName("groupBox_6")
        self.label_12 = QtWidgets.QLabel(self.groupBox_6)
        self.label_12.setGeometry(QtCore.QRect(31, 28, 141, 24))
        self.label_12.setObjectName("label_12")
        self.lnRow = QtWidgets.QLineEdit(self.groupBox_6)
        self.lnRow.setGeometry(QtCore.QRect(70, 30, 31, 21))
        self.lnRow.setObjectName("lnRow")
        self.lnRow.setText(str(ReSCUNeuralNetDialog.step_sz[0]))
        self.lnColumn = QtWidgets.QLineEdit(self.groupBox_6)
        self.lnColumn.setGeometry(QtCore.QRect(180, 30, 31, 21))
        self.lnColumn.setObjectName("lnColumn")
        self.lnColumn.setText(str(ReSCUNeuralNetDialog.step_sz[1]))
        self.label_13 = QtWidgets.QLabel(self.groupBox_6)
        self.label_13.setGeometry(QtCore.QRect(120, 28, 141, 24))
        self.label_13.setObjectName("label_13")
        self.label_13.raise_()
        self.label_12.raise_()
        self.lnRow.raise_()
        self.lnColumn.raise_()

        self.retranslateUi(NNet)
        self.buttonBox.accepted.connect(NNet.accept)
        self.buttonBox.rejected.connect(NNet.reject)
        QtCore.QMetaObject.connectSlotsByName(NNet)

    def retranslateUi(self, CNN):
        _translate = QtCore.QCoreApplication.translate
        CNN.setWindowTitle(_translate("NNet", "Train network"))
        self.groupBox_2.setTitle(_translate("NNet", "Project files"))
        self.label.setText(_translate("NNet", "training image folder"))
        self.btnSavePositive.setText(_translate("NNet", "..."))
        self.groupBox_3.setTitle(_translate("NNet", "Network input size"))
        self.label_4.setText(_translate("NNet", "width"))
        self.label_5.setText(_translate("NNet", "height"))
        self.groupBox_5.setTitle(_translate("NNet", "Other parameters"))
        self.label_10.setText(_translate("NNet", "epochs"))
        self.label_11.setText(_translate("NNet", "learning rate"))
        self.groupBox_6.setTitle(_translate("NNet", "Subimage size (testing)"))
        self.label_12.setText(_translate("NNet", "rows"))
        self.label_13.setText(_translate("NNet", "columns"))
        self.label_14.setText(_translate("NNet", "batch size"))
        self.label_15.setText(_translate("NNet", "erosion width"))
        self.label_16.setText(_translate("NNet", "concatenation level"))
        self.cbGenerateNotebook.setText(_translate("NNet", "generate training notebook"))
        self.btnSaveNotebook.setText(_translate("NNet", "..."))

    def parameters(self) -> dict:
        # os.path.join(ReSCUNeuralNetDialog.positive_training_folder, '') adds a slash at the end of the filename.
        ReSCUNeuralNetDialog.positive_training_folder = os.path.join(self.positive_training_folder_edit.text(), '')

        # Because of the U-Net architecture, input layer dimensions must be divisible by 16 to avoid issues.
        input_size = (int(self.lnHeight.text()), int(self.lnWidth.text()))
        height = (input_size[0] // 16) * 16 if input_size[0] % 16 else input_size[0]
        width = (input_size[1] // 16) * 16 if input_size[1] % 16 else input_size[1]
        ReSCUNeuralNetDialog.train_image_size = (height, width)

        ReSCUNeuralNetDialog.step_sz = (int(self.lnRow.text()), int(self.lnColumn.text()))
        ReSCUNeuralNetDialog.epochs = int(self.lnEpochs.text())
        ReSCUNeuralNetDialog.learning_rate = float(self.lnEta.text())
        ReSCUNeuralNetDialog.mini_batch_size = int(self.lnBatchSz.text())
        ReSCUNeuralNetDialog.erosion_width = int(self.lnErosionWidth.text())
        ReSCUNeuralNetDialog.generate_notebook = self.cbGenerateNotebook.isChecked()
        ReSCUNeuralNetDialog.notebook_path = self.notebook_path_edit.text()
        ReSCUNeuralNetDialog.concatenation_level = int(self.lnConcatenationLevel.text())

        return {
            'positive_training_folder': ReSCUNeuralNetDialog.positive_training_folder,
            'train_image_size': ReSCUNeuralNetDialog.train_image_size,
            'step_sz': ReSCUNeuralNetDialog.step_sz,
            'epochs': ReSCUNeuralNetDialog.epochs,
            'learning_rate': ReSCUNeuralNetDialog.learning_rate,
            'mini_batch_size': ReSCUNeuralNetDialog.mini_batch_size,
            'erosion_width': ReSCUNeuralNetDialog.erosion_width,
            'generate_notebook': ReSCUNeuralNetDialog.generate_notebook,
            'notebook_path': ReSCUNeuralNetDialog.notebook_path,
            'concatenation_level': ReSCUNeuralNetDialog.concatenation_level
        }

    def _open_notebook_path_dialog(self) -> bool:
        start_folder = self.notebook_path_edit.text() if self.notebook_path_edit.text() != '' else self.positive_training_folder_edit.text() if self.positive_training_folder_edit.text() != '' else ReSCUNeuralNetDialog.notebook_path
        folder = RUtils.open_folder_dialog(f"Results folder", start_folder)

        if folder == '' or folder is False:
            return False

        self.notebook_path_edit.setText(folder)

        return True