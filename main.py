from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Templates folder
templates = Jinja2Templates(directory="templates")

# Home route → HTML return karega
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Predict route (HTML form se POST aayega)
@app.post("/predict")
def predict(
    Pregnancies: int = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):
    user_data = np.array([[Pregnancies, Glucose, BloodPressure,
                           SkinThickness, Insulin, BMI,
                           DiabetesPedigreeFunction, Age]])

    prediction = model.predict(user_data)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return {"prediction": result}