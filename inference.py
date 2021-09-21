import numpy as np
import pickle

model = pickle.load(open("C:\\Users\\sys\\DS\\AI_ML_Projects\\Heart failure prediction\\model.pkl","rb"))

def predict(df):
    prediction = model.predict(df)                      # 1 = Heart disease,0= Normal
    pred_prob = Model.predict_proba(df)            # probability of Normal and Heart disease
    return pred_prob,prediction
  