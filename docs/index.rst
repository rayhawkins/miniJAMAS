.. PyJAMAS documentation master file, created by
   sphinx-quickstart on Mon Nov 11 17:25:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

PyJAMAS_ documentation
======================

.. image:: images/paperfigure_v6.png
    :width: 65%
    :align: center

Image analysis has become central for the interpretation and quantification of biological microscopy images. Recent
advances in imaging systems and computer hardware have enabled the application of machine learning to the analysis of
biological images. Python has emerged as the computer language of choice for machine learning and computer vision.
Python is easy to learn for non-computer experts, provides highly optimized numerical packages for machine learning,
image processing and analysis, and supports open source development.

PyJAMAS_ is a Python-based platform for the analysis of microscopy images. PyJAMAS_ can be used for image processing,
object detection—using both machine learning and traditional approaches—, and quantitation of cellular dynamics.
PyJAMAS_ can be extended using plugins written in Python; and the PyJAMAS_ application programming interface enables the
use of PyJAMAS_ in Python programs and scripts.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   manual/installation
   manual/handling_images
   manual/image_annotation
   manual/image_transformations
   manual/watershed
   manual/inflate_balloon
   manual/segmentation_supervised_classifiers
   manual/segmentation_unet
   manual/segmentation_rescunet
   manual/measuring
   manual/plugins
   manual/sampleplugin
   manual/heterogeneityplugin
   manual/api
   manual/citing
   manual/sponsors



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
