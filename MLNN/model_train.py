import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

from MLNN.model_parameters import *

def MLNN_model_train(X_train, Y_train):
    INPUT_SHAPE = X_train.shape[1:]

    model = Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(NUM_HIDDEN_1, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_HIDDEN_2, activation='relu'))
    model.add(Dropout(DROPOUT))

    model.add(Dense(CATEGORIES, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS, verbose=1, validation_split=VALIDATION_SPLIT)

    # Try to create model directory
    try:
        os.makedirs("./model")
    except:
        pass

    # Save the model as h5 file
    model.save("./model/MLNN.h5")


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()
    model_train(X_train, Y_train)