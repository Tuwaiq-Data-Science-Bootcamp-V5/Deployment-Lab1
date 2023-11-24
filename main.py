from fastapi import FastAPI
import pickle
from pydantic import BaseModel


app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))


from pydantic import BaseModel

class Input(BaseModel):
    ph: int
    N: int
    P: int
    K: int


@app.post("/predict")
def predict(input_data: Input):
    data = [[input_data.ph, input_data.N,input_data.P, input_data.K]]
    prediction = model.predict(data)

    return {"prediction": prediction}