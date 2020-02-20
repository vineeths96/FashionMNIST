from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import glorot_normal

from MLNN.model_parameters import *

# Define the model and its structure
def create_model(INPUT_SHAPE):
    model = Sequential()

    model.add(Flatten(input_shape=INPUT_SHAPE))

    model.add(Dense(NUM_HIDDEN_1, kernel_initializer=glorot_normal()))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(Dropout(DROPOUT))

    model.add(Dense(NUM_HIDDEN_2, kernel_initializer=glorot_normal()))
    model.add(LeakyReLU(alpha=ALPHA))
    model.add(Dropout(DROPOUT))
    model.add(Dense(CATEGORIES, activation='softmax'))

    model.summary()

    return model


if __name__ == "__main__":
    create_model()