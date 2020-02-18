from MLNN.load_data import load_data
from MLNN.model_train import model_train
from MLNN.model_test import model_test

X_train, Y_train, X_test, Y_test = load_data()
model_train(X_train, Y_train)
model_test(X_test, Y_test)
