.. _segmentation_supervised_classifiers:

.. _PyJAMAS: https://bitbucket.org/rfg_lab/pyjamas/src/master/

================================================
Object segmentation using linear classifiers
================================================

Supervised machine learning classifiers can detect cellular or subcellular structures with characteristic shape or intensity features. Supervised classifiers must be “trained” to detect the target structure using sample images. Binary classifiers determine whether the target structure is present or not in an image.

The training dataset for supervised, binary classifiers consists of a set of sample images of the structure to be detected (positive training set); and a group of images that does not contain the target structure (negative set). Based on specific features (morphology, brightness, etc.) of the training images, and differences in feature values across the positive and negative datasets, binary classifiers can determine whether the target structure is present or not at each pixel of a test image.

Linear classifiers use linear combinations of features of the structures to be identified to generate a boundary that separates structures belonging to different categories. PyJAMAS_ implements two types of linear classifiers, logistic regression and support vector machines.

Creating training sets
======================

#. Open a grayscale image or time-lapse sequence that includes examples of the structure to be detected (e.g. cell nuclei).

#. Set the working folder using *Set working folder ...* under the **Options** menu. Training images will be automatically saved into the working folder.

#. Draw a rectangle around one of the structures to be segmented. Record the dimensions of the rectangle drawn from the status bar at the bottom of the PyJAMAS_ window. All images in the positive and negative training sets should be of the same size.

#. Save the region of the image within the rectangle using *Save image in polygon ...* from the **IO** menu. The new file name is a combination of the name of the original image, the XY coordinates of the subimage, and the slice number. The image will be saved in the set working directory.

#. Move the rectangle to a different instance of the target structure using the *Move polyline* option from the **Annotations** menu. Click and drag the rectangle to the desired position. The cursors can be used to further refine the position of the rectangle.

#. Repeat steps 3 and 4 until you have a collection of images of the structure to be segmented. These images constitute the positive training set. Make sure that your positive training set represents the diversity of shapes and intensities of the structure to be detected. In general, the more examples, the better. We typically use 100-300 images in our training sets, but we have obtained classification accuracies greater than 90% using 80 images.

#. Create a negative training set by repeating steps 2-6, using images with the same dimensions as the images in the positive training set. To create a negative training set, select image regions that do not contain the structures to be detected.

#. (Optional) Create a training set for hard negative mining, using large images that do not contain the structure to be segmented. Hard negative mining reduces false positives. After training the classifier, images in the hard negative set are scanned to detect the structures to be segmented. As hard negative images do not contain the target structure, any detections are false positives. In such cases, the subregions that were mistakenly labeled as target structures are added to the negative training set, and training is repeated.

Training a classifier
=====================

#. Select a classifier to be trained (logistic regression model or support vector machine) from the **Image**, **Classifiers** menu.

#. Fill in the parameter values to train the selected classifier:

   a. **project files**: include the paths to the folders containing positive, negative, and hard negative (if any) training sets.

   b. **training image size**: the width and height of the images used in the positive and negative sets.

   c. **image step size**: number of pixel rows and columns to skip when scanning test images for target structures. Decreasing the step size leads to increased object detection resolution, at the expense of longer processing times. Increasing the step size accelerates image classification but can reduce its accuracy.

   d. **image features**: select *histogram of gradients*. The histogram of gradients describes the shape of a structure using the distribution of gradient directions in an image of the structure. At each pixel, the gradient or spatial derivative is a vector pointing in the direction of maximum intensity change. The magnitude of the vector corresponds to the magnitude of the intensity change: it is large around edges, and small in flat regions of the image. To calculate the histogram of gradients, the image is divided into non-overlapping windows (in PyJAMAS_, windows are 8x8 pixels), and for the pixels within each window, a histogram of gradient directions is calculated (with 8 bins in PyJAMAS_). To compensate for illumination effects, histograms from window blocks (2x2 windows in PyJAMAS_) are concatenated and contrast-normalized with a measure of intensity calculated over the entire block. The final feature vector for the image consists of the concatenated histograms of all possible blocks. Thus, for example, for a 40x40 image, the feature vector generated by PyJAMAS_ contains:
       8 features ⁄ window x 4 windows ⁄ block x 16 possible blocks = 512 features

   e. **classifier-specific parameters**:

      * The **logistic regression** classifier has a single parameter, the misclassification penalty, a regularization factor that determines the importance of correct classification of all training data. A large misclassification penalty increases the associated cost of classification errors, resulting in a more accurate fitting of the training data, an increased chance of overfitting, i.e. a reduced flexibility of the classifier to detect the target structure in images that deviate somewhat from those represented in the training set.

      * The **support vector machine** (SVM) classifier has two parameters:

        A. **misclassification penalty**: regularization factor that determines the importance of correct classification of all training data. A large misclassification penalty increases the associated cost of classification errors, resulting in a more accurate fitting of the training data, an increased chance of overfitting, i.e. a reduced flexibility of the classifier to detect the target structure in images that deviate somewhat from those represented in the training set.

        B. **kernel type**: *linear* or a *radial basis function* (rbf). A linear SVM defines a linear hyperplane in feature space to separate positive and negative training sets, and applies that hyperplane to classify new samples. In contrast, an rbf kernel transforms the features to an exponential hyperplane, enabling distinction of populations that may not be linearly separable. It is recommended that a linear classifier is first tested, as it is computationally faster and often sufficiently accurate for classification of most data types.

#. Select *OK* and wait for the classifier to be trained. A message on the status bar will indicate training completion.

#. Save the trained classifier using the *Save current classifier ...* option from the **IO** menu.

#. Classifiers can be loaded using the *Load classifier ...* option from the **IO** menu.

Using a classifier
==================

#. To detect structures in an image using a classifier, open the image and make sure to train a classifier or load a trained classifier.

#. Select *Apply classifier ...* from the **Image**, **Classifiers** menu, and choose the slices to apply the classifier to.

#. PyJAMAS_ will add a rectangle annotation around each of the objects detected by the classifier. Each structure will be detected multiple times.

#. Remove redundant annotations using *Non-maximum suppression ...* from the **Image**, **Classifiers** menu. Non-maximum suppression is the process of removing non-optimal detections from the classification results. Three parameters control non-maximum suppression:

   a. **maximum number of objects**: limits the number of target structures present on each slice. The classifiers in PyJAMAS_ assign a probability that each detected object belongs to the class of structures represented by the positive training set. This option selects the maximum number of objects with the highest probabilities and deletes the rest.

   b. **minimum object probability**: sets a minimum threshold for the probability that a detected object belongs to the class of structures represented by the positive training set. Tune the value to prevent false positives.

   c. **maximum intersection/union**: determines the degree of overlap allowed between two rectangle annotations, based on the ratio of the areas of the intersection of the rectangles and their union.  A value of 0.0 will not allow rectangles to overlap with each other, while a value of 1.0 allows rectangles to fully overlap.

#. Rectangle annotations can also be interactively removed.

#. To apply watershed-based segmentation to the detected structures, select *Segment detected objects ...* from the **Image**, **Classifiers** menu. This option will place a seed at the center of the rectangle annotation and segment the object using the watershed algorithm. **Segment detected objects ...** does not include tracking across slices. Use **Track fiducials** to track objects after segmentation.

#. Rectangles bounding structures that are not properly segmented can be removed and segmented using the LiveWire algorithm.
