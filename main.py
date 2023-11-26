from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from log import log
import uvicorn
import pickle
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates=Jinja2Templates(directory="templats")

pickle_in = open("logmodel.pkl", "rb")
logmodel = pickle.load(pickle_in)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("Welcome.html", {"request": request, "data": "my data"})


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse("predict_form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: float = Form(...),
    thal: float = Form(...),
):
    predict = logmodel.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    if int(predict[0]) == 0:
      return templates.TemplateResponse("prediction_result.html", {"request": request, "prediction": "You do not need to go to the hospital"})
    else:
        return templates.TemplateResponse("prediction_result.html", {"request": request, "prediction": "You need to go to the hospital"})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
