pip install fastapi uvicorn scikit-learn 
from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

app = FastAPI()