import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from MLNN.model_parameters import *

def MLNN_load_data():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()

    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255

    X_train = X_train.reshape(X_train.shape[0], PIXELS, PIXELS, 1)
    X_test = X_test.reshape(X_test.shape[0], PIXELS, PIXELS, 1)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    load_data()