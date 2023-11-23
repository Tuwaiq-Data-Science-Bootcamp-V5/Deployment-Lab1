# Deployment-Lab1

### Lab: Deploying a Machine Learning Model with FastAPI

### Obectives:
Your task is to create a FastAPI application that deploys a machine learning model along with a preprocessing step (such as: StandardScaler or any variable that relevant). Additionally, you need to implement a FastAPI router to create both an API and a simple web application that displays predictions.

### Requirements:
#### Use a Pre-trained Model:
- Utilize the machine learning model you trained in the last lab.
#### Implement FastAPI Router for API:
- Create a FastAPI router using the `APIRouter()` function.
#### Define the following endpoint:
- **/:** returns HTMLResponse Welcoming to the website.
- **/api/predict:** Accepts input data and returns the prediction.
- **/predict:** Accepts input data and returns HTMLResponse with results.

#### Implement Web Application:
- - Develop a web application using FastAPI's HTMLResponse.

- - Allow users to input data, submit it, and view the prediction.

- - Use the root path (/) for the home page of the web application.

## Submission:
- You should submit your code and your model and all that included in preprocessing process (such as: `StandardScalar` ..etc).
- Also, You should submit your lab that you choose to deploy.

## Tips:
- Refer to the FastAPI documentation for guidance on routing and HTMLResponse: FastAPI Documentation
- Use the uvicorn command to run your FastAPI application: uvicorn main:app --reload
- Ensure your pre-trained model file is accessible and loaded properly in your FastAPI application.

