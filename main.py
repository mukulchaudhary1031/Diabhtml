from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

with open("model_Rb.pkl", "rb") as f:
    model = pickle.load(f)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Pregnancies: float = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: float = Form(...)
):
    try:
        user_data = pd.DataFrame([[
            Pregnancies, Glucose, BloodPressure,
            SkinThickness, Insulin, BMI,
            DiabetesPedigreeFunction, Age
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ])

        prediction = model.predict(user_data)[0]
        result = "Diabetic 🔴" if prediction == 1 else "Not Diabetic 🟢"

    except Exception as e:
        result = f"Error: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result
    })