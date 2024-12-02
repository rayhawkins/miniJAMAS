"""
    PyJAMAS is Just A More Awesome Siesta
    Copyright (C) 2018  Rodrigo Fernandez-Gonzalez (rodrigo.fernandez.gonzalez@utoronto.ca)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from typing import Optional, Tuple

import skimage.io as sio
import skimage.morphology as sm
import skimage.measure as sme
import skimage.segmentation as ss
import skimage.transform as st
import scipy.ndimage as ndimage

import tensorflow as tf
import tensorflow.keras.backend as kb
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
import tensorflow.keras.utils as ku
import numpy

import pyjamas.pjscore
from pyjamas.rimage.rimml.classifier_types import classifier_types
from pyjamas.rimage.rimutils import rimutils
from pyjamas.rimage.rimml.rimrecurrentneuralnet import rimrecurrentneuralnet
from pyjamas.rutils import RUtils


class ReSCUNet(rimrecurrentneuralnet):

    EROSION_WIDTH: int = 0
    CLASSIFIER_TYPE: str = classifier_types.RESCUNET.value
    CONCATENATION_LEVEL: int = 0
    VALIDATION_SPLIT: float = 0.1

    def __init__(self, parameters: Optional[dict] = None):
        super().__init__(parameters)

        self.W_train: numpy.ndarray = None

        output_classes: int = parameters.get('output_classes', ReSCUNet.OUTPUT_CLASSES)
        learning_rate: float = parameters.get('learning_rate', ReSCUNet.LEARNING_RATE)

        input_size: Tuple[int, int, int] = parameters.get('train_image_size', ReSCUNet.TRAIN_IMAGE_SIZE)

        self.epochs: int = parameters.get('epochs', ReSCUNet.EPOCHS)
        self.mini_batch_size: int = parameters.get('mini_batch_size', ReSCUNet.BATCH_SIZE)

        self.save_folder: str = parameters.get('save_folder')
        self.resize_images_flag: bool = parameters.get('resize_images_flag')
        self.train_network_flag: bool = parameters.get('train_network_flag')
        self.EROSION_WIDTH: int = parameters.get('erosion_width', self.EROSION_WIDTH)
        self.CONCATENATION_LEVEL: int = parameters.get('concatenation_level')

        classifier_representation = parameters.get('classifier')
        if type(classifier_representation) is km.Model:
            self.classifier = classifier_representation
        else:
            if len(input_size) == 2:
                input_size = input_size + (1, )
            self.classifier = self.make_weighted_loss_rescunet(input_size, output_classes)

        adam = ko.Adam(learning_rate=learning_rate)
        self.classifier.compile(adam, loss=self.pixelwise_loss)

        if type(classifier_representation) is list:
            self.classifier.set_weights(classifier_representation)

        parameter_step_sz = parameters.get('step_sz', None)
        if parameter_step_sz is not None and type(parameter_step_sz) == int and parameter_step_sz > 0:
            self.step_sz = (parameter_step_sz, parameter_step_sz)
            print(self.step_sz)
        elif type(parameter_step_sz) == tuple and parameter_step_sz[0] > 0 and parameter_step_sz[1] > 0:
            self.step_sz = parameter_step_sz
        else:
            self.step_sz = ReSCUNet.STEP_SIZE

    def make_weighted_loss_rescunet(self, input_shape: Tuple, n_classes: int) -> km.Model:
        _epsilon = tf.convert_to_tensor(kb.epsilon(), numpy.float32)

        # several input layers for data preprocessing steps
        input_layer = tf.keras.Input(shape=input_shape, name="image_input")

        # the shape of the weight maps has to be such that it can be element-wise
        # multiplied to the softmax output.
        weight_input_layer = tf.keras.Input(shape=input_shape[:2] + (n_classes,))
        mask_input_layer = tf.keras.Input(shape=input_shape, name="mask_input")

        # ensure that the concatenation level is between 0 to 4, inclusive
        if self.CONCATENATION_LEVEL < 0 or self.CONCATENATION_LEVEL > 4:
            print("Invalid concatenation level, setting concatenation level to 0.")
            self.CONCATENATION_LEVEL = 0

        curr_level = 0
        if self.CONCATENATION_LEVEL == curr_level:
            in1 = [kl.Concatenate()([input_layer, mask_input_layer])]
        else:
            in1 = [input_layer]

        # adding the layers; image convolutions
        conv1 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(in1[0])
        conv1 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = kl.Dropout(0.1)(conv1)
        mpool1 = kl.MaxPool2D()(conv1)

        if self.CONCATENATION_LEVEL > curr_level:
            # mask convolutions
            convm1 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mask_input_layer)
            convm1 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convm1)
            convm1 = kl.Dropout(0.1)(convm1)
            mpoolm1 = kl.MaxPool2D()(convm1)

        curr_level += 1
        if self.CONCATENATION_LEVEL == curr_level:
            in2 = [kl.Concatenate()([mpool1, mpoolm1])]
        else:
            in2 = [mpool1]

        conv2 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(in2[0])
        conv2 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = kl.Dropout(0.2)(conv2)
        mpool2 = kl.MaxPool2D()(conv2)

        if self.CONCATENATION_LEVEL > curr_level:
            # mask convolutions
            convm2 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpoolm1)
            convm2 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convm2)
            convm2 = kl.Dropout(0.2)(convm2)
            mpoolm2 = kl.MaxPool2D()(convm2)

        curr_level += 1
        if self.CONCATENATION_LEVEL == curr_level:
            in3 = [kl.Concatenate()([mpool2, mpoolm2])]
        else:
            in3 = [mpool2]

        conv3 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(in3[0])
        conv3 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = kl.Dropout(0.3)(conv3)
        mpool3 = kl.MaxPool2D()(conv3)

        if self.CONCATENATION_LEVEL > curr_level:
            # mask convolutions
            convm3 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpoolm2)
            convm3 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convm3)
            convm3 = kl.Dropout(0.3)(convm3)
            mpoolm3 = kl.MaxPool2D()(convm3)

        curr_level += 1
        if self.CONCATENATION_LEVEL == curr_level:
            in4 = [kl.Concatenate()([mpool3, mpoolm3])]
        else:
            in4 = [mpool3]

        conv4 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(in4[0])
        conv4 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = kl.Dropout(0.4)(conv4)
        mpool4 = kl.MaxPool2D()(conv4)

        if self.CONCATENATION_LEVEL > curr_level:
            convm4 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpoolm3)
            convm4 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(convm4)
            convm4 = kl.Dropout(0.4)(convm4)
            mpoolm4 = kl.MaxPool2D()(convm4)

        curr_level += 1
        if self.CONCATENATION_LEVEL == curr_level:
            in5 = [kl.Concatenate()([mpool4, mpoolm4])]
        else:
            in5 = [mpool4]

        conv5 = kl.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(in5[0])
        conv5 = kl.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = kl.Dropout(0.5)(conv5)

        up6 = kl.Conv2DTranspose(512, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv5)
        conv6 = kl.Concatenate()([up6, conv4])
        conv6 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = kl.Dropout(0.4)(conv6)

        up7 = kl.Conv2DTranspose(256, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
        conv7 = kl.Concatenate()([up7, conv3])
        conv7 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = kl.Dropout(0.3)(conv7)

        up8 = kl.Conv2DTranspose(128, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
        conv8 = kl.Concatenate()([up8, conv2])
        conv8 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = kl.Dropout(0.2)(conv8)

        up9 = kl.Conv2DTranspose(64, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
        conv9 = kl.Concatenate()([up9, conv1])
        conv9 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = kl.Dropout(0.1)(conv9)

        c10 = kl.Conv2D(n_classes, 1, activation='softmax', kernel_initializer='he_normal', name="unet-activation")(
            conv9)

        # Add a few non-trainable layers to mimic the computation of the cross-entropy loss,
        # so that the actual loss function just has to perform the aggregation.
        c11 = kl.Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(c10)
        c11 = kl.Lambda(lambda x: tf.clip_by_value(x, _epsilon, 1. - _epsilon))(c11)
        c11 = kl.Lambda(lambda x: kb.log(x))(c11)
        weighted_sm = kl.multiply([c11, weight_input_layer])

        return km.Model(inputs=[input_layer, weight_input_layer, mask_input_layer], outputs=[weighted_sm])

    def pixelwise_loss(self, target, output):
        """
        A custom function defined to simply sum the pixelwise loss.
        This function doesn't compute the crossentropy loss, since that is made a
        part of the model's computational graph itself.
        Parameters
        ----------
        target : tf.tensor
            A tensor corresponding to the true labels of an image.
        output : tf.tensor
            Model output
        Returns
        -------
        tf.tensor
            A tensor holding the aggregated loss.
        """
        return - tf.reduce_sum(target * output,
                               len(output.get_shape()) - 1)

    def fit(self) -> bool:
        self._process_training_data()

        ycat = ku.to_categorical(self.Y_train)
        wmap = numpy.zeros((self.X_train.shape[0], self.train_image_size[0], self.train_image_size[1], 2),
                           dtype=numpy.float32)
        wmap[..., 0] = self.W_train.squeeze()
        wmap[..., 1] = self.W_train.squeeze()
        self.classifier.fit([self.X_train, wmap, self.X_train_mask], ycat, batch_size=self.mini_batch_size,
                            validation_split=self.VALIDATION_SPLIT, epochs=self.epochs)

        return True

    @classmethod
    def weight_map(cls, binmasks: numpy.ndarray, w0: float = 10., sigma: float = 5.):

        labmasks, n_objs = sm.label(binmasks, return_num=True)

        nrows, ncols = labmasks.shape[:2]
        masks = numpy.zeros((n_objs, nrows, ncols))
        distMap = numpy.zeros((nrows * ncols, n_objs))
        X1, Y1 = numpy.meshgrid(numpy.arange(nrows), numpy.arange(ncols))
        X1, Y1 = numpy.c_[X1.ravel(), Y1.ravel()].T

        for i in range(n_objs):
            mask = numpy.squeeze(labmasks == i + 1)
            masks[i] = mask
            bounds = ss.find_boundaries(mask, mode='inner')
            X2, Y2 = numpy.nonzero(bounds)
            xSum = (X2.reshape((-1, 1)) - X1.reshape((1, -1))) ** 2
            ySum = (Y2.reshape((-1, 1)) - Y1.reshape((1, -1))) ** 2
            distMap[:, i] = numpy.sqrt(xSum + ySum).min(axis=0)

        ix = numpy.arange(distMap.shape[0])
        if distMap.shape[1] == 1:
            d1 = distMap.ravel()
            border_loss_map = w0 * numpy.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
        else:
            if distMap.shape[1] == 2:
                d1_ix, d2_ix = numpy.argpartition(distMap, 1, axis=1)[:, :2].T
            else:
                d1_ix, d2_ix = numpy.argpartition(distMap, 2, axis=1)[:, :2].T
            d1 = distMap[ix, d1_ix]
            d2 = distMap[ix, d2_ix]
            border_loss_map = w0 * numpy.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))

        xBLoss = numpy.zeros((nrows, ncols))
        xBLoss[X1, Y1] = border_loss_map
        # class weight map
        loss = numpy.zeros((nrows, ncols))
        w_1 = 1 - masks.sum() / loss.size
        w_0 = 1 - w_1
        loss[masks.sum(0) == 1] = w_1
        loss[masks.sum(0) == 0] = w_0
        ZZ = xBLoss + loss
        # ZZ = resize(ZZ, outsize, preserve_range=True)

        return ZZ

    def _process_training_data(self) -> bool:
        """
        Loads and scales training data and calculates weight maps.
        :return:
        """
        train_ids = next(os.walk(self.positive_training_folder))[1]

        # Get and resize train images and masks
        self.X_train = numpy.zeros((len(train_ids), self.train_image_size[0], self.train_image_size[1], 1), dtype=numpy.uint16)
        self.X_train_mask = numpy.zeros((len(train_ids), self.train_image_size[0], self.train_image_size[1], 1),
                                        dtype=bool)
        self.Y_train = numpy.zeros((len(train_ids), self.train_image_size[0], self.train_image_size[1], 1), dtype=bool)
        self.W_train = numpy.zeros((len(train_ids), self.train_image_size[0], self.train_image_size[1], 1), dtype=float)
        print('Getting and resizing train images and masks ... ')

        for n, id_ in enumerate(train_ids):
            print(f"Image number: {n}/{len(train_ids)}")
            path = os.path.join(self.positive_training_folder, id_)
            im_file = os.path.join(path, "image", os.listdir(os.path.join(path, "image"))[0])
            img = sio.imread(im_file)
            if img.ndim == 3:
                img = img[0, :, :]
            img = numpy.expand_dims(st.resize(img, (self.train_image_size[0], self.train_image_size[1]), order=3,
                                              mode='constant', preserve_range=True), axis=-1)
            self.X_train[n] = img

            prev_mask_file = os.path.join(path, "prev_mask", os.listdir(os.path.join(path, "prev_mask"))[0])
            prev_mask = numpy.zeros((self.train_image_size[0], self.train_image_size[1], 1), dtype=bool)
            prev_mask_ = sio.imread(prev_mask_file)
            prev_mask_ = numpy.expand_dims(st.resize(prev_mask_, (self.train_image_size[0], self.train_image_size[1]), order=3,
                                                     mode='constant', preserve_range=True), axis=-1)
            # Resizing interpolates values that makes the mask not binary, fix this by dividing by the max value,
            # rounding, and then multiplying again by the max value
            max_value = numpy.max(numpy.max(prev_mask_))
            prev_mask_ = numpy.round(numpy.divide(prev_mask_, numpy.full(prev_mask_.shape, max_value)))
            prev_mask_ = numpy.multiply(prev_mask_, numpy.full(prev_mask_.shape, max_value))
            prev_mask = numpy.maximum(prev_mask, prev_mask_)
            self.X_train_mask[n] = prev_mask

            msk_file = os.path.join(path, "mask", os.listdir(os.path.join(path, "mask"))[0])
            mask = numpy.zeros((self.train_image_size[0], self.train_image_size[1], 1), dtype=bool)
            mask_ = sio.imread(msk_file)
            mask_ = numpy.expand_dims(st.resize(mask_, (self.train_image_size[0], self.train_image_size[1]), order=3,
                                                mode='constant', preserve_range=True), axis=-1)

            # Resizing interpolates values that makes the mask not binary, fix this by dividing by the max value,
            # rounding, and then multiplying again by the max value
            max_value = numpy.max(numpy.max(mask_))
            mask_ = numpy.round(numpy.divide(mask_, numpy.full(mask_.shape, max_value)))
            mask_ = numpy.multiply(mask_, numpy.full(mask_.shape, max_value))
            mask = numpy.maximum(mask, mask_)
            self.Y_train[n] = mask
            weights = ReSCUNet.weight_map(mask)
            self.W_train[n, :, :, 0] = weights

        self.scaler = numpy.amax(self.X_train)
        self.X_train = self.X_train / self.scaler

        return True

    def predict(self, image: numpy.ndarray, prev_mask: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
        if image is None or image is False:
            return False
        if prev_mask is None or prev_mask is False:
            return False

        if image.ndim == 3:
            image = image[0, :, :]
        if prev_mask.ndim == 3:
            prev_mask = prev_mask[0, :, :]

        testImage = image / self.scaler

        image_input = self.classifier.get_layer('image_input').input
        prev_mask_input = self.classifier.get_layer('mask_input').input
        softmax_output = self.classifier.get_layer('unet-activation').output
        predictor = kb.function([image_input, prev_mask_input], [softmax_output])

        testLabel = numpy.zeros(testImage.shape, dtype=bool)
        testProb = numpy.zeros(testImage.shape, dtype=bool)
        half_width = int(self.train_image_size[1] / 2)
        half_height = int(self.train_image_size[0] / 2)
        for animage, therow, thecol in rimutils.generate_subimages(testImage, self.train_image_size[0:2],
                                                                   self.step_sz, True):
            prev_mask_subimage = prev_mask[(therow - half_height):(therow + half_height),
                                           (thecol - half_width):(thecol + half_width)]
            yhat = predictor([numpy.expand_dims(animage, axis=0), numpy.expand_dims(prev_mask_subimage, axis=0)])[0]
            yhat = numpy.argmax(yhat[0], axis=-1)
            p = numpy.amax(yhat[0], axis=-1)

            testLabel[(therow - half_height):(therow + half_height), (thecol - half_width):(thecol + half_width)] \
                = numpy.logical_or(testLabel[(therow - half_height):(therow + half_height),
                                   (thecol - half_width):(thecol + half_width)], yhat)
            testProb[(therow - half_height):(therow + half_height), (thecol - half_width):(thecol + half_width)] = p

        # perform postprocessing to fill holes and take only the one object that best matches the previous mask
        testLabel = ndimage.binary_fill_holes(testLabel)
        labelled_mask = sme.label(testLabel)
        region_ids = numpy.unique(labelled_mask[labelled_mask > 0])
        if len(region_ids) > 1:
            best_IOU = None
            best_id = None
            for this_id in region_ids:
                this_region = labelled_mask == this_id
                intersection = numpy.sum(numpy.logical_and(this_region, prev_mask))
                union = numpy.sum(numpy.logical_or(this_region, prev_mask))
                this_IOU = intersection / union
                if best_id is None or this_IOU > best_IOU:
                    best_IOU = this_IOU
                    best_id = this_id
            testLabel = labelled_mask == best_id

        if self.EROSION_WIDTH is not None and self.EROSION_WIDTH != 0:
            self.object_array = numpy.asarray(rimutils.extract_contours(
                sm.dilation(sm.label(sm.binary_erosion(testLabel, sm.square(self.EROSION_WIDTH)), connectivity=1),
                            sm.square(self.EROSION_WIDTH))), dtype=object)
        else:
            self.object_array = numpy.asarray(rimutils.extract_contours(sm.label(testLabel, connectivity=1)),
                                              dtype=object)
        self.prob_array = testProb

        return self.object_array.copy(), self.prob_array.copy()

    def save(self, filename: str) -> bool:
        classifier = self.classifier.get_weights() if self.classifier.weights else self.classifier

        theclassifier = {
            'classifier_type': self.CLASSIFIER_TYPE,
            'positive_training_folder': self.positive_training_folder,
            'train_image_size': self.train_image_size,
            'scaler': self.scaler,
            'epochs': self.epochs,
            'mini_batch_size': self.mini_batch_size,
            'learning_rate': kb.eval(self.classifier.optimizer.lr),
            'classifier': self.classifier.get_weights(),
            'step_sz': self.step_sz,
            'erosion_width': self.EROSION_WIDTH,
            'concatenation_level': self.CONCATENATION_LEVEL
        }

        return RUtils.pickle_this(theclassifier, RUtils.set_extension(filename, pyjamas.pjscore.PyJAMAS.classifier_extension))
