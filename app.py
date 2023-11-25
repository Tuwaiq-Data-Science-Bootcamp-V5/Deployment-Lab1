import pickle
from fastapi import FastAPI, APIRouter, Form
from fastapi.responses import HTMLResponse

app = FastAPI()
router = APIRouter()

# Load your pre-trained model and preprocessing steps using pickle
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@router.get("/", response_class=HTMLResponse)
async def input_data():
    return '''
    <h1> Soil Prediction! </h1>
        <form method="post">
        N: <input type="number" step="0.01" name="N" placeholder="Nitrogen"><br>
        P: <input type="number" step="0.01" name="P" placeholder="Phosphorous"><br>
        K: <input type="number" step="0.01" name="K" placeholder="Potassium"><br>
        pH: <input type="number" step="0.1" name="pH" placeholder="pH"><br>
        <button type="submit">Predict</button>
        </form>
    '''

@router.post("/", response_class=HTMLResponse)
async def predict(N: float = Form(...), P: float = Form(...), K: float = Form(...), pH: float = Form(...)):
    data = [N, P, K, pH]  # Create a list of input values
    prediction = model.predict([data])  # Make prediction
    return f'''
        <h2>Prediction Result:</h2>
        <p>Input Data: {data}</p>
        <p>Prediction: {prediction[0]}</p>
        <a href="/">Go Back</a>
    '''

app.include_router(router)