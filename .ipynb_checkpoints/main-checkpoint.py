import pickle
import uvicorn
from fastapi import FastAPI
from  Crop import Crop


## loading the saved model and the scaler
model = pickle.load(open("KNN.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = FastAPI()

@app.post("/predict")
def predict(crop: Crop):
    name = ['apple', 'banana', 'rice', 'pomegranate', 'pigeonpeas', 'papaya',
             'orange', 'muskmelon', 'mungbean', 'mothbeans', 'mango', 'maize', 'lentil',
               'kidneybeans', 'jute', 'grapes', 'cotton', 'coffee', 'coconut', 'chickpea', 'blackgram', 'watermelon']
    code = [0, 1, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 21]
    
    crop.crop = crop.crop.lower()
    if(crop.crop == "male"):
        crop.crop = 0
    else:
        crop.crop = 1


    X = [[
        crop.N,
        crop.P,
        crop.K,
        crop.ph,
    ]]


    X = scaler.transform(X)
    Y = model.predict(X)[0]

    return {
        "Crop": int(Y)
    }