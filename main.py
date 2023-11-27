from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import StandardScaler

class InputData(BaseModel):
    sepal_length: float
    sepal_width :float
    petal_length :float
    petal_width :float

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="templates")
pickle_in = open('model.pkl','rb')
model = pickle.load(pickle_in)
preprocessor = StandardScaler()

@app.get("/")
def greeting():
    return {"message" : "welcome to my model"}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@router.post("/api/predict")
def predict(data : InputData):
    input_dict = data.dict()
    input_features = [input_dict['sepal_length'],input_dict['sepal_width'],input_dict['petal_length'],input_dict['petal_width']]
    input_array = preprocessor.transform([input_features])
    prediction = model.predict(input_array)[0]
    return {"prediction": prediction}

app.include_router(router)

    