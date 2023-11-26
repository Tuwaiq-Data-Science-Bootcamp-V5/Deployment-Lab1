from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import uvicorn

app = FastAPI()

pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

class Linear(BaseModel):
    ENGINESIZE: float

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post('/predict')
def predict_species(data: Linear):
    data = data.dict()
    ENGINESIZE = data['ENGINESIZE']    
    weight = model.coef_
    bias = model.intercept_
    pred = ENGINESIZE * weight[0] + bias
    return {'Pred is': str(pred[0])}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
