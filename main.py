from fastapi import FastAPI, Form, Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
from pydantic import BaseModel

class Soil(BaseModel):
    N: int
    P: int
    K: int
    ph: float
   

async def model(soil : Soil):
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    decoder = pickle.load(open("decoder.pkl", "rb"))

    X = [[
        soil.N,
        soil.P,
        soil.K,
        soil.ph
    ]]
    X = scaler.transform(X)
    Y = model.predict(X)[0] 
    return decoder[Y]



app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("welcome.html",
                                    {"request": request}
                                    )

@router.get("/predict")
async def predict_html(request: Request):
    return templates.TemplateResponse("form.html",
                                    {"request": request}
                                    )

@router.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, N: int = Form(...), P: int = Form(...), K: int = Form(...), ph: float = Form(...)):
    soil = Soil(N=N, P=P, K=K, ph=ph) 
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "crop": await model(soil)},
    )

@router.post("/api/predict")
async def predict(soil: Soil):
    prediction = await model(soil)
    return {
        "crop": prediction
    }



app.include_router(router)
