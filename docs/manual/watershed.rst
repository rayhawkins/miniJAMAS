.. _watershed:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

==================================
Watershed-based image segmentation
==================================

The watershed algorithm is a region-growing method useful to segment touching objects. Briefly, the watershed requires identification of one –and only one– seed point per object to be segmented. The algorithm simulates a flooding process on a topographical surface in which the height of each pixel corresponds to its value. Flooding begins at the seed points, and pixels are assigned to the seed from which water reached them. Watershed lines (edges) are built at pixels where water from two seed points meets. To use the watershed algorithm in PyJAMAS_:

#. Automatically find seeds for the objects to be segmented (e.g. cells) using the *Find seeds ...* option from the **Image** menu. In the dialog, adjust the parameters to optimize the seeds detected, and click *OK* to place the seeds on the image.

   The automated seed detection method in PyJAMAS_ applies adaptive thresholding to binarize the image. An adaptive threshold uses a different pixel value to distinguish foreground from background at each pixel, in contrast to a global threshold, which uses the same threshold value throughout the image. In PyJAMAS_, the adaptive threshold is the mean pixel value within a window around each pixel. Discontinuities in the resulting binary image are resolved using mathematical morphology operations that first dilate (thicken) and then erode (thin) bright features in the image. The resulting image is inverted, and a distance transform is used to identify pixels within the cytoplasm of each cell. In the distance transform, the value of each pixel corresponds to its distance to the closest background pixel. The local maxima of the distance transform define one point per cell. There are four parameters in the *Find seeds ...* dialog to optimize seed detection:

   a. **smoothing σ**: kernel size for a Gaussian filter to make the background uniform and blur image boundaries.

   b. **local window size**: window size used to calculate adaptive thresholds.

   c. **binary closings**: determines the radius of the disk used to perform morphological filtering on the image. Use values of the approximate size of the discontinuities in the features to segment. If the value is too large, independent objects will be merged. If segmenting bright edges on a dark background (e.g. fluorescently-labeled cell outlines) use positive values. If segmenting dark edges on a bright background (e.g. differential interference contrast images of cells), or bright objects separated by dark pixels (e.g. fluorescently-labeled cell nuclei), use negative radii.

   d. **minimum distance to background**: minimum distance transform value for a pixel to be considered to be inside the object to be segmented. The value is used to calculate the local maxima of the distance transform: the centers of each group of connected pixels after thresholding the distance transform with this value are used as seeds.

   On the right side of the *Find seeds ...* dialog, the results of smoothing, thresholding, and distance transform calculations are displayed.

    .. image:: ../images/find_seeds_cells.gif
        :width: 75%
        :align: center

   When *Find seeds ...* is applied to several slices, the seeds on each slice are not necessarily added in the same order, and therefore the fiducial ids for the same object will vary across slices. PyJAMAS_ can be used to track fiducial ids across slices using *Track fiducials ...* in the **Annotations** menu. Alternatively, fiducials identified for one slice can be automatically propagated to other slices using *Propagate seeds ...* or *Expand and propagate seeds ...* from the **Image** menu.

    .. image:: ../images/track_seeds_cells.gif
        :width: 75%
        :align: center

   *Expand and propagate seeds ...* enables the use of one set of seeds to segment multiple images. After identifying seeds in a single slice (manually or using *Find seeds ...*) and expanding the seeds using the watershed method, the algorithm detects fiducials too close to the boundary of the object they belong to, and moves them to the centroid of the polyline outlining the object. All fiducials are then projected onto the next slice. Seed projection uses the cross-correlation between the source and the target images. Cross-correlation is a mathematical tool to calculate similarity between two signals. Briefly, the source image is divided into windows, and each window is scanned over the target image, multiplying overlapping pixels at each possible position of the source window over the target image. Similarity between the two images will lead to greater values of the sum of pixel products. The window size is determined by the **xcorr window** parameter in the *Expand and propagate seeds ...* dialog. Note that it is also possible to project seeds onto subsequent time points without expanding them, using the *Propagate seeds ...* option in the **Image** menu. In this case, seeds are not centered before projecting them onto the next slice. Seed propagation can be used to automatically generate fiducials for image registration.

    .. image:: ../images/expand_propagate_cells.gif
        :width: 75%
        :align: center

    Another alternative to identify a consistent set of seeds is to use *Propagate seeds ...*. After identifying seeds in a single slice (manually or using *Find seeds ...*), seeds are projected onto the next slice using the cross-correlation between the source and the target images. No boundary-proximity correction is applied, as the cells are not segmented.

    .. image:: ../images/propagate_seeds_cells.gif
        :width: 75%
        :align: center

#. One fiducial will be added to the image representing each one of the seeds. Fiducials should be edited to ensure the presence of one and only one seed per cell.

#. Select *Expand seeds ...* from the **Image** menu to segment object using the watershed algorithm.

    .. image:: ../images/expand_seeds_cells.gif
        :width: 75%
        :align: center

#. Select *Expand and propagate seeds ...* from the **Image** menu to segment objects on the current slice using the watershed algorithm, and to automatically project the seeds onto the next slice. This option integrates object segmentation and tracking, and it ensures that corresponding objects will be annotated with the same fiducial and polyline ids.

#. The *LiveWire* can be used to interactively correct automated segmentation results. Incorrect polylines can be deleted by selecting any of the polyline annotation modes under the **Annotations** menu (rectangle, polyline or LiveWire), and right-clicking on the appropriate polyline.

   Fiducial and polyline ids are assigned independently of each other. Manual correction of segmentation polylines will result in mismatched fiducial and polyline ids. Use *Track fiducials ...* in the **Annotations** to match fiducial ids.