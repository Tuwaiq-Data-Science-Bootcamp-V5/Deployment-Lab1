from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import StandardScaler


class Data(BaseModel):
     N: int
     P: int
     K: int
     ph: float 

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="templates")

model = joblib.load("\Users\rifal\Desktop\KNN-Lab1\Assignment.ipynb")
preprocessor = StandardScaler() 

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("input.html", {"request": request})


@router.post("/api/predict")
def predict(data: Data):
    input_dict = data.dict()
    input_features = [input_dict["N"], input_dict["P"],  input_dict["K"],  input_dict["ph"]]  
    input_array = preprocessor.transform([input_features])
    prediction = model.predict(input_array)[0]
    return {"prediction": prediction}


app.include_router(router)
