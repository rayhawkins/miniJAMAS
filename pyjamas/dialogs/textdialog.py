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

from PyQt6 import QtCore, QtWidgets


class TextDialog(object):
    def __init__(self, text: str = '', title: str = ''):
        super().__init__()
        self.dialog = QtWidgets.QDialog()
        self.setupUi(self.dialog)
        self.textBrowser.setText(text)
        self.dialog.setWindowTitle(title)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(653, 459)
        Dialog.setWindowFlags(QtCore.Qt.WindowType.CustomizeWindowHint)  # Disable maximize, minimize, close buttons - this last one can block mouse interaction with the main window.
        self.okButton = QtWidgets.QPushButton(Dialog)
        self.okButton.setGeometry(QtCore.QRect(520, 420, 113, 32))
        self.okButton.setObjectName("okButton")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(20, 20, 611, 381))
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Dialog)
        self.okButton.clicked.connect(Dialog.close)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.okButton.setText(_translate("Dialog", "Cool!"))

    def show(self):
        self.dialog.exec()
        self.dialog.show()

    def resize(self, w: int, h: int):
        self.dialog.resize(w, h)
