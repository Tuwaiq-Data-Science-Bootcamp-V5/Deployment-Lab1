from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
from pydantic import BaseModel

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def base():
    pass


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