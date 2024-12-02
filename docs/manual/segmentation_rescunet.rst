.. _segmentation_rescunet:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

.. _Colab: https://colab.research.google.com/

.. _U-Net: https://arxiv.org/abs/1505.04597

.. _ReSCU-Net: https://bitbucket.org/raymond_hawkins_utor/rescu-net/src/main/

================================================
Object segmentation using ReSCU-Nets
================================================

ReSCU-Net_ is a deep learning architecture based on the U-Net_ and specialized for interactive segmentation of multi-dimensional data.

To segment an object in an image, the ReSCU-Net_ uses the image data, as well as the segmentation mask of that object from the previous timepoint.

The training dataset consists of a set of images of the objects to be detected (cells, nuclei, etc.), corresponding binary image masks labelling the objects in the input images, and binary image masks labelling the same objects in the previous timepoint. During training, a loss function measures how far the network prediction is from the target segmentation. The loss function is weighed with a map that has greater values at pixels in the proximity of the edges of structures. Increasing the weight of edge pixels penalizes segmentation errors at boundaries, ensuring that the network can correctly find object boundaries.


Creating training sets
======================

#. Outline the objects to be detected by the classifier. This can be done using any of the tools (rectangles, polylines, LiveWires) available in PyJAMAS_, in the entire image or in subregions thereof. When outlining objects, ensure that each object is drawn in the same order on each frame so that the object ids are unchanged throughout timepoints. (Object ids can be shown by toggling *Display fiducial and polyline ids* in the **Options** menu).

#. Using your segmentations create a training dataset in a parent folder where each subfolder corresponds to a specific object at a specific timepoint. Each subfolder should have folders for the image at the current timepoint, object mask for the current timepoint, and object mask for the previous timepoint. If there are multiple objects segmented throughout the video, this can be done by exporting each object into its own annotation file using **I/O** > *Export individual fiducial-polyline annotations*, followed by saving the annotations as masks for each slice using **I/O** > *Export current ROIs as binary image*. The files should then be rearranged into the appropriate folder structure, as shown below.

        .. image:: ../images/rescunet_file_structure.png
            :width: 75%
            :align: center


Training a ReSCU-Net
====================

#. Select *Create and train ReSCUNet ...* from the **Image**, **Classifiers** menu.

#. Fill in the parameter values to train the ReSCU-Net_:

   a. **training image folder**: path to the folder containing the training set.

   b. **network input size**: the width and height of the images that will be fed into the network. Training images will be rescaled to this size. Because of the architecture of the network, the selected dimensions must be divisible by 16 (but not necessarily equal to each other). Smaller input images generate smaller networks with fewer parameters that train faster. However, smaller networks are worse at resolving boundaries between touching structures. 32x32, 64x64, 128x128 or 192x192 are typical values.

   c. **subimage size (testing)**: number of pixel rows and columns to divide test images into (post training, when the networks is applied to new images). Each subimage will be scaled to the network input size and processed. Decreasing the subimage size leads to increased object detection resolution, at the expense of longer processing times. Increasing the step size accelerates image classification but can reduce its accuracy.

   d. **learning rate**: scale factor that affects the step size when minimizing the cost function of the network. Larger values lead to faster training, with the possibility of missing cost function minima. Smaller values are more likely to converge to the minimum of the cost function, but take longer to get there.

   e. **batch size**: number of images in the training set that are propagated through the network before updating the weights. Smaller values result in a noisy minimization of the cost function and slower learning, but the trained networks are more generalizable. A typical value is 32.

   f. **epochs**: number of times that the entire training data set will be run through the network.

   g. **concatenation level**: the number of encoder blocks that the image and previous mask are processed separately before being combined. A concatenation level of 1 works well for most datasets, but higher concatenation levels can improve accuracy if objects change a lot between timepoints (max. 4). Increasing concatenation level increases the size of the network, leading to longer training and prediction times.

   h. **erosion width**: size of the erosion kernel applied to the binary image produced by the trained network when applied to a new image. For U-Net_, the erosion width is used to separate nearby objects after prediction. Since ReSCU-Net_ predicts each object one at a time, objects do not need to be separated so an erosion width of 0 is recommended.

   i. **generate training notebook**: training a neural network is computationally expensive. If you are not running the training in a computer equipped with a Graphics Processing Unit (GPU), it may take a long time for training to finish. Thus, PyJAMAS_ offers the possibility of generating a notebook that can be uploaded together with the training data to platforms such as Colab_ for faster training. Colab_ offers free remote access to GPU-equipped machines. When executed on Colab_, the notebook generated by PyJAMAS_ will train the network and save a model that can be used in Colab_ or downloaded and loaded into PyJAMAS_ for application to new images. Check this box to generate the notebook (saved at the path indicated on the textbox to the right) and train remotely, or leave unchecked to run the training on the local computer.

#. Select *OK* and wait for the network to be trained. A message on the status bar will indicate training completion.

#. Save the trained network using the *Save current classifier ...* option from the **IO** menu. Or run the notebook in Colab_ and download the trained networks.

#. Trained networks can be loaded using the *Load classifier ...* option from the **IO** menu.


Training a ReSCU-Net in Colab
=============================

#. When you create the ReSCU-Net_, make sure to check **generate training notebook**.

#. In the selected folder, a new file with .ipynb extension (a notebook) will be created.

#. Upload the notebook and both train and test data to Colab_. Alternatively, upload the train and test data to your google drive, open the training notebook in colab, and mount your google drive to Colab_ (allowing you to access your data without having to upload it each time you open Colab_). The notebook assumes that both train and test data will be uploaded in a zip file named testtrain.zip. But this is easy to edit in the notebook. It is important to store each training image in an independent folder, each of which contains three subfolders: *image*, *mask*, and *prev_mask*, that in turn contain the current image frame, the binary mask highlighting the object in the current frame, and the binary mask highlighting the object in the previous frame, respectively.

#. Make sure that your connection is to a runtime equipped with a GPU (you can validate this with the *Change runtime* option under the **Runtime** menu).

#. In Colab_, run through the notebook. Training will take some time. When training is done, make sure to download the generated model.


Using a ReSCU-Net
=================

#. To detect structures in an image using a ReSCU-Net_, open the image and make sure to train a network or load a trained network.

#. Outline the objects in the first timepoint that you wish to segment.

#. Move to the next image frame.

**Option 1: segment entire videos at once** (works well if you have few objects on each frame, or a network with very high accuracy)

4. Select *Apply classifier ...* from the **Image**, **Classifiers** menu, and choose the slices to apply the network to.

5. PyJAMAS_ will add a polyline annotation around each of the objects detected by the classifier.

6. If the network performs poorly on some slices, you can correct the erroneous segmentation for the first incorrect slice, then reapply the network for the following slices.


        .. image:: ../images/rescunet_prediction_whole_video.gif
            :width: 75%
            :align: center

**Option 2: segment videos frame-by-frame** (works well if you have many objects, or objects that are difficult for the network to segment)

4. Select *Apply classifier to current slice ...* from the **Image**, **Classifiers** menu (or use the Shift+A shortcut).

5. Correct any erroneous segmentations produced by the network.

6. Advance to the next slice and repeat.


        .. image:: ../images/rescunet_prediction_framebyframe.gif
            :width: 75%
            :align: center
