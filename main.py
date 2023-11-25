from fastapi import FastAPI, Form, Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
from pydantic import BaseModel



# Initialize FastAPI app
app = FastAPI()

# Initialize APIRouter
api_router = APIRouter()

# Mount static files
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define a Pydantic model for input data
class InputData(BaseModel):
    N: int
    P: int
    K: int
    ph: float
    # Add other features as needed

async def model(input : InputData):
   with open('KNN.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    X = [[
        input.N,
        input.P,
        input.K,
        input.ph
    ]]
    X = scaler.transform(X)
    Y = model.predict(X)[0] 
    return Y

# Define API routes
@api_router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("welcome.html",
                                    {"request": request}
                                    )
@api_router.post("/api/predict")
async def predict_api(data: InputData):
    prediction = await model(data)
    return {
        "crop": prediction
    }

@api_router.post("/predict")
async def predict(request: Request, N: int = Form(...), P: int = Form(...), K: int = Form(...), ph: float = Form(...)):
    data = InputData(N=N, P=P, K=K, ph=ph) 
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "crop": await model(data)},
    )

@api_router.get("/predict")
async def predict_html(request: Request):
    return templates.TemplateResponse("form.html",
                                    {"request": request}
                                    )

# Include the APIRouter in the main app
app.include_router(api_router)

# Run the app using Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
