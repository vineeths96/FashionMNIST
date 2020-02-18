import os

os.environ['TF_XPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

from CNN.model_parameters import *

def CNN_model_train(X_train, X_test):
    INPUT_SHAPE = X_train.shape[1:]

    model = Sequential()

    model.add(Conv2D(NUM_FILTERS_1, kernel_size=KERNAL_SIZE_1, input_shape=INPUT_SHAPE, padding='same', activation='relu'))
    model.add(MaxPool2D(POOL_SIZE_1))

    model.add(Flatten())
    model.add(Dense(NUM_DENSE_1, activation='relu'))
    model.add(Dense(CATEGORIES, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    history = model.fit(x=X_train, y=X_test, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS, validation_split=VALIDATION_SPLIT)

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