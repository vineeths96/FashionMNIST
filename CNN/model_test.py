import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from CNN.model_parameters import *

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from analytics.plot_confusion_matrix import plot_confusion_matrix


def CNN_model_test(X_test, Y_test, network):
    output_file = open('convolution-neural-net.txt', 'w')

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

    confusion_mtx = confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(confusion_mtx, network, classes=range(10))

    target_names = ["Class {}".format(i) for i in range(CATEGORIES)]
    classification_rep = classification_report(Y_test, Y_pred, target_names=target_names, output_dict=True)

    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True)
    plt.savefig('./results/' + network + '_classification_report.png')
    #plt.show()


if __name__ == "__main__":
    model_test()
