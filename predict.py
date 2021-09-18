
import pickle
import numpy as np

model = pickle.load(open("/Users/prathiksingh/Desktop/srikanth/model.pkl"))

def predict(x_test_csv,y_test_csv):
    prediction = model.predict(x_test_csv)
    accuracy = sklearn.metrics.accuracy_score(prediction,y_test_csv)
    return accuracy

  