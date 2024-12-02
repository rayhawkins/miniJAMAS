.. _installation:

.. _anaconda: https://www.anaconda.com/
.. _homebrew: https://brew.sh/
.. _ipywidgets: https://ipywidgets.readthedocs.io/en/latest/user_install.html
.. _Jupyter: https://jupyter.org/
.. _mamba: https://mamba.readthedocs.io/en/latest/installation.html
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Nodejs: https://nodejs.org/
.. _pip: https://pip.pypa.io
.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/
.. _PyQt6: https://www.riverbankcomputing.com/software/pyqt/
.. _pypi: https://pypi.org
.. _Python: https://www.python.org/downloads/
.. _venv: https://docs.python.org/3/library/venv.html
.. _virtualenv: https://virtualenv.pypa.io/

============
Installation
============

The `app <https://www.quantmorph.ca/software/pyjamas-python>`_
==============================================================

The easiest way to install PyJAMAS_ is using the app for MacOS-Intel, MacOS-Silicon and Windows 11, which can be downloaded `here <https://www.quantmorph.ca/software/pyjamas-python>`_. **HOWEVER**, if you want to take advantage of machine learning models and GPU capabilities on MacOS or Windows machines, you should use the pypi_ installation as described below.

**Known problems**:

#. *Corrupt application / unidentified developer*

In MacOS, depending on the security settings, running the app for the first time can cause an error message indicating that the application is broken or from an unidentified developer. Check `this site <https://support.apple.com/en-ca/guide/mac-help/mh40616/mac#:~:text=Control%2Dclick%20the%20app%20icon,you%20can%20any%20registered%20app.>`_ for ways to allow PyJAMAS_ to run. Alternatively, open a Terminal and type:

.. code-block:: bash

 $ xattr -cr /Applications/pyjamas.app

substituting "/Applications/pyjamas.app" with the path where PyJAMAS_ was installed. The same trick can be applied to the dmg file before installation:

.. code-block:: bash

 $ xattr -cr pyjamas.dmg


Using pypi_
==========================

Follow the steps below to install PyJAMAS_ from the pypi_ repository.

**APPLE-INTEL**

#. Install Python_ 3.10.

#. Download and install PyJAMAS_ by opening a terminal and typing:

   .. code-block:: bash

    $ python -m pip install pyjamas-rfglab

**APPLE-SILICON**

#. Install Python_ 3.10.

#. Download and install PyJAMAS_:

   .. code-block:: bash

    $ python -m pip install pyjamas-rfglab

#. PyJAMAS_ supports the use of the GPU on the Silicon chips. You just need to install tensorflow-metal:

   .. code-block:: bash

    $ python -m pip install tensorflow-metal

**Known problems**:

#. PyJAMAS_ uses PyQt6_ to create a cross-platform user interface, but PyQt6_ does not play well with MacOS Catalina or earlier MacOS versions. If you are using an old MacOS and cannot upgrade, you can still use the last version of PyJAMAS_ written with PyQt5:

   .. code-block:: bash

    $ python -m pip install pyjamas-rfglab==2023.8.0

**WINDOWS**

#. Install Python_ 3.10.

#. Download and install PyJAMAS_ by opening a terminal and typing:

   .. code-block:: bash

    $ python -m pip install pyjamas-rfglab

#. PyJAMAS_ supports the use of CUDA-enabled GPUs in Windows. Please, check here (https://tensorflow.org/install/gpu) for instructions on how to configure your system. Briefly:

   a. Download and install the NVIDIA GPU drivers (https://www.nvidia.com/drivers).

   b. Download and install the CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit-archive).

   c. Download and install the cuDNN SDK (https://developer.nvidia.com/cudnn and https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

**Known problems**:

#. CUDA and cuDNN are picky with the version of each other that they talk to. If PyJAMAS_ displays an error that cusolver64_10.dll is not found:

   a. Go to the folder C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\V11.2\\bin (replacing V11.2 by whichever version you installed).

   b. Create a copy of the file cusolver64_11.dll.

   c. Rename the copy as cusolver64_10.dll.

#. Import skimage can cause the following error: "ImportError: DLL load failed while importing _rolling_ball_cy: The specified module could not be found.". To fix the error:

   a. The error is caused because the system is missing the Microsoft C and C++ runtime libraries. These libraries are required by many applications built by using Microsoft C and C++ tools.

   b. Download and install the Microsoft C and C++ runtime libraries by running vcredist_x64.exe (https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

**LINUX**

#. Install Python_ 3.10.

#. Download and install PyJAMAS_ by opening a terminal and typing:

   .. code-block:: bash

    $ python -m pip install pyjamas-rfglab

**Known problems**:

#. In Linux systems, you will need permissions to install PyJAMAS_ globally. To restrict the PyJAMAS_ installation to the current user, install it with:

   .. code-block:: bash

    $ python –m pip install --user --no-cache-dir -U pyjamas-rfglab

**RUNNING PyJAMAS**

#. To run PyJAMAS_, open a terminal and type:

   .. code-block:: bash

    $ pyjamas

#. The code for PyJAMAS_ can be found in the PyJAMAS_ folder under the Python_ site packages (e.g. /usr/local/lib/python3.10/site-packages in MacOS). The location of the source code is important to extend PyJAMAS_ using `plugins <plugins.html>`_. Alternatively, you can download the PyJAMAS_ source code from: https://bitbucket.org/rfg_lab/pyjamas/src/master/.

#. PyJAMAS_ can be run from the source code by opening a terminal, navigating to the folder that contains the code, and typing:

   .. code-block:: bash

    $ python -m pyjamas.pjscore

**Known problems**:

#. The analysis of image batches in PyJAMAS_ can generate interactive Jupyter_ notebooks. Interactivity in Jupyter_ notebooks relies on ipywidgets_, a package installed with PyJAMAS_. Please, check the ipywidgets_ documentation if you have issues with interactivity in notebooks (e.g. there are no interactive features). Most often the following steps are sufficient to fix any issues:

   a. Download and install the Nodejs_ JavaScript runtime.

   b. Open a new terminal and execute the following command for JupyterLab:

   .. code-block:: bash

    $ jupyter labextension install @jupyter-widgets/jupyterlab-manager

   or this one for Jupyter Notebook:

   .. code-block:: bash

    $ jupyter nbextension enable --py widgetsnbextension

   c. Reopen your Jupyter_ server.

Using virtual environments
==========================

Virtual environments allow isolation of Python_ packages, preventing interference and incompatibilities between different package dependencies and versions thereof.

PyJAMAS_ can be used inside a virtual environment. We strongly recommend the use of conda_ environments.

*conda*
*******

Download and install anaconda_, miniconda_ or mamba_. In a terminal or an anaconda_ power shell, create a virtual environment with:

.. code-block:: bash

 $ conda create -n envpyjamas python=3.10

substituting *3.10* with the version of the Python_ interpreter that you would like to use, and *envpyjamas* with the name of the virtual environment. If you use mamba_, use *mamba* instead of *conda*.

anaconda_ stores virtual environments within the folder that contains the anaconda_ distribution. You can find the location of your virtual environment with:

.. code-block:: bash

 $ conda info --envs


Next, activate the environment with:

.. code-block:: bash

 $ conda activate envpyjamas

Now you may proceed with the download and installation of PyJAMAS_ as above, using *python* as the name of the Python_ interpreter.

You can deactivate the virtual environment at any time with:

.. code-block:: bash

 $ conda deactivate
