.. _image_annotation:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

================
Image annotation
================

#. PyJAMAS_ provides four tools to manually annotate images for subsequent quantification:

   a. **Fiducials.** Fiducials are small, magenta circles, stored as the x- and y-coordinates of their centre. Fiducials can be added by selecting *Fiducials* from the **Annotations** menu. Any subsequent clicks on the left mouse button will add a fiducial. To remove a fiducial, right click on it.

        .. image:: ../images/fiducials.gif
            :width: 75%
            :align: center

   b. **Rectangles.** Rectangles are polylines (see below), represented with a green outline, and stored as an object containing the x- and y-coordinates of the four corners that make up the rectangle. Rectangles can be added by selecting *Rectangles* from the **Annotations** menu. To add a rectangle, click on the position of the first corner and drag the mouse to the position of the opposite corner before releasing. The dimensions of the rectangle (in pixels) are displayed on the status bar at the bottom of the PyJAMAS_ window. To remove a rectangle, right click anywhere on the boundary or inside the rectangle.

        .. image:: ../images/rectangles.gif
            :width: 75%
            :align: center

   c. **Polylines.** Polylines are polygons, represented with a green outline, and stored as an object containing each vertex of the polygon. Polylines can be added by selecting *Polylines* from the **Annotations** menu. Single clicks on the left mouse button will place the first and each subsequent vertex in a polyline; a double-click on the left mouse button will close the polyline. To remove a polyline, right click anywhere on the boundary or inside the polyline.

        .. image:: ../images/polylines.gif
            :width: 75%
            :align: center

   d. **LiveWire.** The LiveWire is a semi-automated method for image segmentation based on Dijsktra’s minimal path search algorithm (35). The LiveWire connects subsequent pixels in a polyline by finding the minimal cost path between the two pixels. In our implementation, the cost of each pixel is the inverse of the pixel value (36). Thus, the algorithm favours paths with high pixel values, which often correspond to cell outlines in fluorescence microscopy images of membrane markers. The LiveWire produces polylines such as the ones described above. However, LiveWire annotations contain a much higher density of points along the perimeter of the object, as they represent a quasi-continuous path. To use the LiveWire tool for feature delineation, select *LiveWire* from the **Annotations** menu. Use a left mouse click to start the LiveWire, then move the mouse along the edge to be delineated. Once you are satisfied with a segment, click on the left mouse button to store that segment. Use a double-click on the left mouse button to close the polyline.

        .. image:: ../images/livewire.gif
            :width: 75%
            :align: center

#. If you press Shift while double-clicking to close a polyline or a LiveWire, the annotation will be terminated at that point and stored as an open polyline.

    .. image:: ../images/open_polylines.gif
        :width: 75%
        :align: center

#. If you press Alt while using the LiveWire, a straight line will be drawn.
    .. image:: ../images/livewire_straight.gif
        :width: 75%
        :align: center

#. Annotations can be toggled on/off using the *Hide/display annotations* option from the **Annotations** menu.

        .. image:: ../images/hide_annotations.gif
            :width: 75%
            :align: center

#. Fiducials and polylines are assigned sequential identification numbers (ids) as they are created. Ids are independent for fiducials and polylines, and can be displayed using the *Display fiducial and polyline ids* option in the **Options** menu. Ids are important to track structures across image sequences: corresponding fiducials and polylines are expected to have matching ids. Ids are reassigned when an annotation is deleted to maintain id continuity within the image.

        .. image:: ../images/ids_annotations.gif
            :width: 75%
            :align: center

#. Polylines can be copied, pasted and moved using the corresponding options under the **Annotations** menu.

        .. image:: ../images/edit_polylines.gif
            :width: 75%
            :align: center

#. Annotations can be reverted using the *Undo* option under the **Options** menu.

#. It is possible to use polylines to generate binary masks (with the pixels inside the polyline and under the edge set to True, and those outside set to False) using *Export current ROIs as binary image ...* from the **IO** menu.

#. To save image annotations, use the *Save annotations ...* option from the **IO** menu.

#. Annotations can be loaded using the *Load annotations ...* option from the **IO** menu. Previously existing annotations will be erased.

   If you use *Load annotations (additive) ...* the new annotations in the selected file will be added to those already existing on the open image. No annotations will be erased.

#. Under the **Annotations** menu, PyJAMAS_ provides options to delete all the annotations on a slice in an image sequence or on the entire image sequence; as well as the annotations within a user-defined polyline.

        .. image:: ../images/delete_annotations.gif
            :width: 75%
            :align: center

