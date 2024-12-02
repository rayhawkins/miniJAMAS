.. _sampleplugin:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

=============
Sample plugin
=============

This plugin displays a hello message both on the status bar of the PyJAMAS_ window and in the terminal.

.. code-block:: python

 from pyjamas.rplugins.base import PJSPluginABC

 # PyJAMAS plugins inherit from the abstract class
 # pyjamas.rplugins.base.PJSPluginABC,
 # and must implement name and run methods.
 class PJSPlugin(PJSPluginABC):
     # This method returns the name of the menu item used to
     # launch the plugin.
     def name(self) -> str:
         return "Sample plugin"

     # This method implements the functionality of the plugin.
     def run(self, parameters: dict) -> bool:
         print("Hello my friends!")
         self.pjs.statusbar.showMessage("Hello my friends!")

         return True