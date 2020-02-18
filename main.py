from MLNN.load_data import MLNN_load_data
from MLNN.model_train import MLNN_model_train
from MLNN.model_test import MLNN_model_test

from CNN.load_data import CNN_load_data
from CNN.model_train import CNN_model_train
from CNN.model_test import CNN_model_test
"""
X_train, Y_train, X_test, Y_test = load_data()
model_train(X_train, Y_train)
model_test(X_test, Y_test)
"""

X_train, Y_train, X_test, Y_test = CNN_load_data()
CNN_model_train(X_train, Y_train)
CNN_model_test(X_test, Y_test)
