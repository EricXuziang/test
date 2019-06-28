import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class predict_unet:
    def __init__(self):
       self.path='model/unet.h5' 
    def mean_iou(y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def dice_coef(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def generalized_dice_coeff(y_true, y_pred):
        Ncl = y_pred.shape[-1]
        w = K.zeros(shape=(Ncl,))
        w = K.sum(y_true, axis=(0,1,2))
        w = 1/(w**2+0.000001)
        # Compute gen dice coef:
        numerator = y_true*y_pred
        numerator = w*K.sum(numerator,(0,1,2,3))
        numerator = K.sum(numerator)
        denominator = y_true+y_pred
        denominator = w*K.sum(denominator,(0,1,2,3))
        denominator = K.sum(denominator)
        gen_dice_coef = 2*numerator/denominator
        return gen_dice_coef

    def generalized_dice_loss(y_true, y_pred):
        return 1 - generalized_dice_coeff(y_true, y_pred)

    def segmentation(self,image):
        
        self.model=load_model(self.path ,custom_objects={'generalized_dice_loss': generalized_dice_loss,'mean_iou':mean_iou,'dice_coef':dice_coef})
        self.file_path='image/'+img_name
        self.img=utils.load_img(self.file_path,target_size=(224,224))
        self.img=image.img_to_array(self.img)
        self.img = self.img.astype('float32')
        self.img /= 255
        self.img_mask=model.predict(self.img,verbose=1)
        self.img_mask = array_to_img(self.img_mask)
		
        self.img_mask.save("image/mask.jpg")


if __name__ == "__main__":
    m = predict_unet()
    m.generalized_dice_loss()
    m.mean_iou()
    m.dice_coef()
    m.segmentation("image/1.jpg")