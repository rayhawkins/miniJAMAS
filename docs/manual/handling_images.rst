.. _handling_images:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

===============
Handling images
===============

#. To **open an image** (including multipage tiff files, png, jpeg, bmp or gif), use the *Load image ...* option from the **IO** menu.

   If you open an image with color slices, you will be asked to open all the slices for one channel or all the channels for one slice. The image will be grayscale.

#. To **save the image currently open** (not including annotations, but including any changes to the pixel values), use the *Save image ...* option from the **IO** menu.

#. To **save the current image, including annotations**, use *Save display ...* (single image) or *Export movie with annotations ...* (all images in a sequence) from the **IO** menu.

#. To **save a region of interest**, draw a polyline (e.g. a rectangle) around that region and select **save the current image, including annotations**, use *Save image in polyline to working folder ...*. To set the working folder, see *Set working folder ...* under the **Options** menu.

#. To **display information about the image** (size and display settings, including minimum and maximum percentiles, zoom, brush size, frames per second, current slice and number of annotations), use  *Display info* in the **Image** menu.

    .. image:: ../images/info.gif
        :width: 75%
        :align: center

#. To **adjust the image contrast**, use  *Adjust contrast* in the **Image** menu and select the pixel value percentiles to map to black (minimum) and white (maximum).

    .. image:: ../images/adjust_contrast.gif
        :width: 75%
        :align: center

#. To **display XZ and YZ slices** of three-dimensional images, use  *Orthogonal views* in the **Image** menu.

    .. image:: ../images/orthogonal.gif
        :width: 75%
        :align: center

#. PyJAMAS_ provides options to **rotate, flip, invert, and project images, or to play time-lapse sequences**, using the corresponding options under the **Image** menu.

    .. image:: ../images/rotate.gif
        :width: 49%

    .. image:: ../images/flip.gif
        :width: 49%

    .. image:: ../images/max_project.gif
        :width: 49%

    .. image:: ../images/sum_project.gif
        :width: 49%

    .. image:: ../images/invert.gif
        :width: 49%

    .. image:: ../images/play.gif
        :width: 49%

#. **Image operations can be reverted** using the *Undo* option under the **Options** menu.