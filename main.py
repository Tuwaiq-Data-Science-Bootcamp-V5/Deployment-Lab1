
import pickle
import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse

#my data class
from  UserInfo import UserInfo

## loading the saved model and the scaler
model = pickle.load(open("KNN.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


def make_predction(N, P, K, PH):
    name = ['apple', 'banana', 'rice', 'pomegranate', 'pigeonpeas', 'papaya',
             'orange', 'muskmelon', 'mungbean', 'mothbeans', 'mango', 'maize', 'lentil',
               'kidneybeans', 'jute', 'grapes', 'cotton', 'coffee', 'coconut', 'chickpea', 'blackgram', 'watermelon']
    code = [0, 1, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 21]


    X = [[N, P, K, PH]]

    X = scaler.transform(X)
    Y = model.predict(X)[0]

    for i in range(0, 22):
        if code[i] == Y:
            Y = name[i]
    
    return Y


app = FastAPI()
router = APIRouter()

############################
####     API predict    ####
############################

@router.post("/api/predict", tags=["API"])
def predict(data: UserInfo, type ='API'):

    Y = make_predction(data.N, data.P, data.K, data.ph)
    getcontent(Y)

    return {
        "Crop": Y,
    }


app.include_router(router)

############################
######  WelcomePage  #######
############################

def generate_html_response():
    html_content = """
    <html>
        <head>
                <title>HI Dear user</title>
        </head>
        <body>
            <h1> This website predict the crop based on your input data </h1>
            <h2> please provide to the predict API these info : </h2>
            <h3> N: Nitrogen content ratio in the soil **int** </h3>
            <h3> P: Phosphorous content ratio in the soil **int** </h3>
            <h3> K: Potassium content ratio in the soil **int** </h3>
            <h3> PH: ph value of the soil **float** </h3>
            <h2> return Value : </h2>
            <h2> - Crop Name </h2>

        </body>
    </html>"""
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/", tags=["WebApp"], response_class=HTMLResponse)
def HomePage():
    return generate_html_response()

############################
####   Web get predict  ####
############################

def getcontent(Y):
    predction = """
    <html>
        <head>
                <title>Dear user</title>
        </head>
        <body>
            <h1> The Prediction value you got from using the API is:  </h1>
            <h2> {Y} </h2>
        </body>
    </html>"""
    return HTMLResponse(content=predction, status_code=200)


@app.get("/predict", tags=["WebApp"], response_class=HTMLResponse)
def predict():
    f = open("D:\\DataScience\\Deployment-Lab1\\response.txt" , "r")
    Y = f.read()
    return HTMLResponse(content=Y, status_code=200)

############################
####  Web Post predict  ####
############################

@app.post("/predict", tags=["WebApp"], response_class=HTMLResponse)
def predict(data: UserInfo, type ='WebGet'):
    Y = make_predction(data.N, data.P, data.K, data.ph)
    getcontent(Y)
    return HTMLResponse(content=Y, status_code=200)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)