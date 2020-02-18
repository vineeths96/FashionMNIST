import os
import numpy as np

os.environ['TF_XPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import glorot_normal

from CNN.model_parameters import *

def CNN_model_train(X_train, Y_train):
    INPUT_SHAPE = X_train.shape[1:]

    X_train = (X_train - np.mean(X_train)) / np.std(X_train)

    datagen = ImageDataGenerator(rotation_range=0, zoom_range=0.01, width_shift_range=0.025, \
                                 height_shift_range=0.025, shear_range=0.01, validation_split=VALIDATION_SPLIT)
    datagen.fit(X_train)

    model = Sequential()

    model.add(Conv2D(NUM_FILTERS_1, kernel_size=KERNAL_SIZE_1, kernel_initializer= glorot_normal(), input_shape=INPUT_SHAPE, padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(POOL_SIZE_1, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(NUM_FILTERS_2, kernel_size=KERNAL_SIZE_2, kernel_initializer= glorot_normal(), input_shape=INPUT_SHAPE, padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(POOL_SIZE_2, STRIDE, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(NUM_FILTERS_2, kernel_size=KERNAL_SIZE_2, kernal_initialiser= glorot_normal(), input_shape=INPUT_SHAPE, padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(POOL_SIZE_2, STRIDE, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Flatten())
    model.add(Dense(NUM_DENSE_1, activation='relu'))
    model.add(Dense(CATEGORIES, activation='softmax'))

    model.summary()

    train_data_generated = datagen.flow(x=X_train, y=Y_train, batch_size=BATCH_SIZE, subset='training')
    validation_data_generated = datagen.flow(x=X_train, y=Y_train, batch_size=BATCH_SIZE, subset='validation')

    STEPS_EPOCH = len(X_train)*(1-VALIDATION_SPLIT)//BATCH_SIZE

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(train_data_generated, steps_per_epoch=STEPS_EPOCH, epochs=TRAINING_EPOCHS,\
                        validation_data=validation_data_generated, validation_steps=int(STEPS_EPOCH*VALIDATION_SPLIT),\
                        callbacks=[learning_rate_reduction])

    # Try to create model directory
    try:
        os.makedirs("./model")
    except:
        pass

    # Save the model as h5 file
    model.save("./model/CNN.h5")


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    model_train(X_train, Y_train)