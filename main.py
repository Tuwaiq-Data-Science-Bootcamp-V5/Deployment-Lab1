from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import StandardScaler


class InputData(BaseModel):
    age: float
    bmi: float   
    
app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="templates")

model = joblib.load("/Users/nadam/Desktop/Deployment-Lab1/ML.ipynb")
preprocessor = StandardScaler() 

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/api/predict")
def predict(data: InputData):
    input_dict = data.dict()
    input_features = [input_dict["age"], input_dict["bmi"]]  
    input_array = preprocessor.transform([input_features])
    prediction = model.predict(input_array)[0]
    return {"prediction": prediction}


app.include_router(router)

#uvicorn main:app --reload