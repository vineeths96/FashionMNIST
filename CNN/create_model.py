from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, InputLayer
from tensorflow.keras.initializers import glorot_normal

from CNN.model_parameters import *

# Define the model and its structure
def create_model(INPUT_SHAPE):
    model = Sequential()

    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(BatchNormalization(axis=1))

    model.add(Conv2D(NUM_FILTERS_1, kernel_size=KERNAL_SIZE_1, kernel_initializer=glorot_normal(), padding='same'))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(POOL_SIZE_1, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(NUM_FILTERS_2, kernel_size=KERNAL_SIZE_2, kernel_initializer=glorot_normal(), padding='same'))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(POOL_SIZE_2, STRIDE, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(NUM_FILTERS_3, kernel_size=KERNAL_SIZE_3, kernel_initializer=glorot_normal(), padding='same'))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(MaxPool2D(POOL_SIZE_3, STRIDE, padding='same'))
    model.add(Dropout(DROPOUT))

    model.add(Flatten())
    model.add(Dense(NUM_DENSE_1, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_DENSE_2, activation='relu'))
    model.add(Dense(CATEGORIES, activation='softmax'))


    model.summary()

    return model


if __name__ == "__main__":
    create_model()