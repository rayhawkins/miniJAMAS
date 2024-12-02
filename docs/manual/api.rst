.. _api:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

================
The PyJAMAS_ API
================

The PyJAMAS_ API can be invoked by creating a PyJAMAS object:

.. code-block:: python

   from pyjamas.pjscore import PyJAMAS
   pjs = PyJAMAS()

The **pjs** object contains a set of attributes that provide access to the PyJAMAS_ API.
The attributes are instances of different submodules in the pyjamas.rcallbacks package.
The attributes are:

- `pjs.io`_
- `pjs.options`_
- `pjs.image`_
- `pjs.classifiers`_
- `pjs.annotations`_
- `pjs.measurements`_
- `pjs.batch`_
- `pjs.plugins`_
- `pjs.about`_

The methods included in each attribute are described below.

pjs.io
======

.. autoclass:: pyjamas.rcallbacks.rcbio.RCBIO
   :members:

pjs.options
===========

.. autoclass:: pyjamas.rcallbacks.rcboptions.RCBOptions
   :members:

pjs.image
=========

.. autoclass:: pyjamas.rcallbacks.rcbimage.RCBImage
   :members:

pjs.classifiers
===============

.. autoclass:: pyjamas.rcallbacks.rcbclassifiers.RCBClassifiers
   :members:

pjs.annotations
===============

.. autoclass:: pyjamas.rcallbacks.rcbannotations.RCBAnnotations
   :members:

pjs.measurements
================

.. autoclass:: pyjamas.rcallbacks.rcbmeasure.RCBMeasure
   :members:

pjs.batch
=========

.. autoclass:: pyjamas.rcallbacks.rcbbatchprocess.RCBBatchProcess
   :members:

pjs.plugins
===========

.. autoclass:: pyjamas.rcallbacks.rcbplugins.RCBPlugins
   :members:

pjs.about
=========

.. autoclass:: pyjamas.rcallbacks.rcbabout.RCBAbout
   :members:
