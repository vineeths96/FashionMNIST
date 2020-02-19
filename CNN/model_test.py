import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def CNN_model_test(X_test, Y_test):
    output_file = open('convolution-neural-net.txt', 'w')

    #X_test = (X_test - np.mean(X_test))/np.std(X_test)

    try:
        model = tf.keras.models.load_model('./model/CNN.h5')
    except:
        print("Trained model does not exist. Please train the model.\n")
        exit()

    loss, accuracy = model.evaluate(x=X_test, y=Y_test)
    Y_pred = model.predict_classes(X_test)

    output_file.write("Loss on Test Data : {}\n".format(loss))
    output_file.write("Accuracy on Test Data : {}\n".format(accuracy))
    output_file.write("gt_label,pred_label\n")
    for idx in range(len(Y_test)):
        output_file.write("{},{}\n".format(Y_test[idx], Y_pred[idx]))

    output_file.close()


if __name__ == "__main__":
    model_test()