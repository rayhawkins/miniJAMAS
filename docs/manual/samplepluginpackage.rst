.. _samplepluginpackage:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

=====================
Sample plugin package
=====================

This example demonstrates a more complex plugin that includes multiple functions callable from a dropdown in the plugins menu.

.. code-block:: python

 from pyjamas.rplugins.base import PJSPluginABC

 # Useful imports if the plugin requires a sub-menu with multiple options.
 from PyQt6 import QtWidgets, QtCore
 from functools import partial

 # PyJAMAS plugins inherit from the abstract class
 # pyjamas.rplugins.base.PJSPluginABC,
 # and must implement name and run methods.
 # They may also optionally implement a
 # build_menu method.
 class PJSPlugin(PJSPluginABC):
     # This method returns the name of the menu item used to
     # launch the plugin.
     def name(self) -> str:
         return "Sample plugin package"

    # This method implements the functionality of the plugin.
     def run(self, parameters: dict) -> bool:
         print(parameters['text'])
         self.pjs.statusbar.showMessage(parameters['text'])

         return True

     # This method defines how the plugin sub-menu looks
     # and what options it contains. It does not need to
     # be implemented if there is only one option.
     def build_menu(self):
         parent_menu = QtWidgets.QMenu(self.pjs.menuPlugins)
         parent_menu.setObjectName(self.name())
         parent_menu.setTitle(QtCore.QCoreApplication.translate('PyJAMAS', self.name()))

         self.pjs.addMenuItem(parent_menu, "Print 'Hello my friends!'", None,
                              partial(self.run, {'text': "Hello my friends!"}))
         self.pjs.addMenuItem(parent_menu, "Print 'Howdy y'all!'", None,
                              partial(self.run, {'text': "Howdy y'all!"}))

         self.menu = self.pjs.menuPlugins.addMenu(parent_menu)