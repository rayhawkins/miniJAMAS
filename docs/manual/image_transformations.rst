.. _image_transformations:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

=====================
Image transformations
=====================

Processing the open image
=========================

#. To crop an image, draw a rectangle around the region of the image to be preserved, and select *Crop* from the **Image** menu. Then click on the rectangle containing the region to be cropped.

   It is possible to crop an image using any type of polyline: the bounding box of the polyline will be used if the polyline is not rectangular.

        .. image:: ../images/crop.gif
            :width: 75%
            :align: center

#. To rescale an image, select *Rescale image ...* from the **Image** menu, and select the scaling factor for the rows (Y-axis) and columns (X-axis).

        .. image:: ../images/rescale.gif
            :width: 75%
            :align: center

#. To create a kymograph, draw a rectangle around the region of the image to be included, and select *Kymograph* from the **Image** menu. Then click on the rectangle containing the region to include in the kymograph.

   It is possible to generate a kymograph using any type of polyline: the bounding box of the polyline will be used if the polyline is not rectangular.

        .. image:: ../images/kymograph.gif
            :width: 75%
            :align: center

#. To register the slices (Z planes or consecutive time points) that form an image sequence, add a constant number of fiducials to the slices to be registered, making sure that the fiducial ids of corresponding structures match. Move to the slice that will be the reference for image registration and select the *Register* option from the **Image** menu.

        .. image:: ../images/register.gif
            :width: 75%
            :align: center

   To speed up the generation of fiducials for image registration, add fiducials to image features on the first image (manually or using *Find seeds ...*). Project the fiducials onto subsequent time points using the *Propagate seeds ...* option in the **Image** menu.

        .. image:: ../images/propagate_seeds.gif
            :width: 75%
            :align: center

   The registration algorithm in PyJAMAS_ will shift images by the mean x and y displacements of the fiducials with respect to the corresponding fiducials in the reference image.

   If fiducial ids do not match, PyJAMAS_ provides a tool to automatically track fiducials and match their ids. Select *Track fiducials ...*  under the **Annotations** menu. For each fiducial in the source slice, the algorithm will determine the closest fiducial in the target (next) slice. If two fiducials from the source slice are mapped on to the same fiducial in the target slice, PyJAMAS_ will produce an error indicating the ids of the overlapping fiducials. The same number of fiducials should be present on each slice for this simple tracking algorithm to work.

        .. image:: ../images/track_fiducials.gif
            :width: 75%
            :align: center

#. To apply **Gaussian smoothing** to an image, select *Gaussian smoothing ...* from the **Image** menu and choose the standard deviation value for the Gaussian kernel.

        .. image:: ../images/gaussian_smoothing.gif
            :width: 75%
            :align: center

#. To calculate the **magnitude of the image gradient**, select *Gradient* from the **Image** menu.

        .. image:: ../images/gradient.gif
            :width: 75%
            :align: center

Batch correct and crop
======================

#. Select the *Correct images ...* option from the **Batch** menu. In the dialog, specify the parameters to correct the images in the selected folder. The different options in the *Correct images ...* dialog are:

   a. **input folder**: path to the folder containing all the images to be resized; can contain subfolders.

        .. image:: ../images/before_ffc.png
            :width: 75%
            :align: center

   b. **darkfield image**: path to the image to be used for dark field correction; this image will be subtracted from the original during the correction procedure, and it is supposed to represent the image of an empty field with no illumination.

        .. image:: ../images/dark_field_image.png
            :width: 75%
            :align: center

   c. **flatfield image**: path to the image to be used for flat field correction; this image will be divided from the dark-field-corrected image and should therefore also be darkfield corrected; it is supposed to represent the image of an empty field with the same illumination used for sample acquisition (typically it is an average of many of these images).

        .. image:: ../images/ffc_image.png
            :width: 75%
            :align: center

   d. **crop dimensions**: images will be cropped to this number of rows and this number of columns before the correction procedure; cropping is also applied to the dark field and flat field images if using; dimensions that are too small will be left intact.

   e. **background**: select *none* for no additional background subtraction, or *mode* to subtract the image mode--calculated for each image, including all its slices--after dark and flat field corrections.

   f. **input substring**: a substring to select which images should be corrected.

   g. **output file suffix**: a suffix added to the file name containing the corrected image; the file name will be the same as for the original image except for this suffix.

#. The corrected image files are saved in the folder containing the corresponding original image file.

        .. image:: ../images/after_ffc.png
            :width: 75%
            :align: center