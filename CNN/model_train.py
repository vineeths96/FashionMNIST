import os

os.environ['TF_XPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

from CNN.create_model import create_model
from CNN.model_parameters import *
from analytics.plot import plot

def CNN_model_train(X_train, Y_train):
    INPUT_SHAPE = X_train.shape[1:]

    # Image data augmentation
    datagen = ImageDataGenerator(zoom_range=0.01, width_shift_range=0.025, height_shift_range=0.025, shear_range=0.01,\
                                 validation_split=VALIDATION_SPLIT, fill_mode='nearest')
    datagen.fit(X_train)

    model = create_model(INPUT_SHAPE)

    # Generate augmented data on the fly
    train_data_generated = datagen.flow(x=X_train, y=Y_train, batch_size=BATCH_SIZE, subset='training')
    validation_data_generated = datagen.flow(x=X_train, y=Y_train, batch_size=BATCH_SIZE, subset='validation')
    STEPS_EPOCH = len(X_train)*(1-VALIDATION_SPLIT)//BATCH_SIZE
    STEPS_VALIDATION = int(STEPS_EPOCH*VALIDATION_SPLIT)

    # Set the optimiser values
    optimizer = RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

    # Train the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(train_data_generated, steps_per_epoch=STEPS_EPOCH, epochs=TRAINING_EPOCHS,\
                        validation_data=validation_data_generated, validation_steps=STEPS_VALIDATION,\
                        callbacks=[learning_rate_reduction])

    # Uncomment for generating plots.
    """
    plot(history, "CNN")
    """

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