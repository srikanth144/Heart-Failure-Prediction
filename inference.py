import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))
y_pred = []
def predict(df):
    df = pd.get_dummies(df,columns = ['Sex','ExerciseAngina','ChestPainType','RestingECG','ST_Slope'])
    names = df.columns
    df = sklearn.preprocessing.minmax_scale(df,feature_range=(0,1))
    df = pd.DataFrame(df,columns=names)
    prediction = model.predict(df)
    pred_prob = model.predict_proba(df) 
    strng = " the probability % of this person not having heart problem is  "
    strng1 = "the probability % of this person  having heart problem is "  
    for j in range(len(prediction)):
        if pred_prob[j][0]>pred_prob[j][1]:
            i = strng+str(pred_prob[j][0]*100)
            y_pred.append(i)
        else:
            i = strng1+str(pred_prob[j][1]*100)
            y_pred.append(i)
    return y_pred