import argparse

from MLNN.load_data import MLNN_load_data
from MLNN.model_train import MLNN_model_train
from MLNN.model_test import MLNN_model_test

from CNN.load_data import CNN_load_data
from CNN.model_train import CNN_model_train
from CNN.model_test import CNN_model_test

arg_parser = argparse.ArgumentParser(description= "Test model (default) or train model ")
arg_parser.add_argument("--train-model", action="store_true", default=False)

argObj = arg_parser.parse_args()

# If argument is present train the network, else test the network and record accuracy
if argObj.train_model == False:
    X_train, Y_train, X_test, Y_test = MLNN_load_data()
    MLNN_model_test(X_test, Y_test, "MLNN")

    X_train, Y_train, X_test, Y_test = CNN_load_data()
    CNN_model_test(X_test, Y_test, "CNN")
else:
    X_train, Y_train, X_test, Y_test = MLNN_load_data()
    MLNN_model_train(X_train, Y_train)

    X_train, Y_train, X_test, Y_test = CNN_load_data()
    CNN_model_train(X_train, Y_train)