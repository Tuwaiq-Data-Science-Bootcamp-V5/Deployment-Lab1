# Deployment-Lab1

### Assignment: Deploying a Machine Learning Model with FastAPI

### Objective:
Your task is to create a FastAPI application that deploys a machine learning model along with a preprocessing step (such as: StandardScaler). Additionally, you need to implement a FastAPI router to create both an API and a simple web application that displays predictions.

### Requirements:

Use a pre-trained machine learning model that you use in last lab.
Implement a FastAPI router for the API with the following endpoints:
/predict: Accepts input data and returns the prediction.

Also, Implement a web application using FastAPI's HTMLResponse that allows users to input data, submit it, and view the prediction.
Use StandardScaler() from scikit-learn for preprocessing before making predictions.

- FastAPI (Home Page: root) /
  - [API](#api) (APIs: with link /api)
  - - predict (/api/predict)
  - [webapp](#webapp) (HTMLResponse with root link)
  - - predict (/predict)

