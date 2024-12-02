.. _inflate_balloon:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

.. _balloon: https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_geodesic_active_contour

================================
Balloon-based image segmentation
================================

A balloon_ (a.k.a. active contour or snake) is a region-growing method useful to segment objects with bright boundaries. Briefly, a balloon_ begins as a small polygon (or even a point) defined inside the object to be segmented. The polygon grows under the influence of an inflation force, while trying to minimize an energy functional that depends on the value of the underlying pixels and the shape of the polygon. The energy of the balloon_ is minimized over bright pixels and when the shape is smooth and convex. To use a balloon_ in PyJAMAS_:

#. Make sure that the objects to be segmented in your image have bright boundaries (*Invert image* and *Gradient* under the **Image** menu can be useful for this).

#. Activate the balloon mode using the *Balloon* option from the **Annotations** menu.

#. Click the mouse inside the object to be segmented. Maintain the mouse pressed to increase the inflation force (you will see the force value in the status bar below your image).

#. Release the mouse:

   - if the object is properly segmented, you are done;

   - if no segmentation appears, you need a stronger inflation force, try again;

   - if the segmentation is not correct, you can delete it by right clicking on it, or you can inflate it by clicking inside.

#. If you would like to use the same inflation force for other objects, click inside the next object and release the mouse while pressing *Shift*. This will automatically apply the last recorded inflation force to the new balloon.

    .. image:: ../images/inflate_balloon.gif
        :width: 75%
        :align: center
