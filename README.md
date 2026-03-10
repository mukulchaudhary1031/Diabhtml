

 Diabetes Prediction Project (with Live Deployment)


# Diabetes Prediction System 🩺

A machine learning project to **predict diabetes** based on patient health data using Python, Random Forest, FastAPI, and Docker.  
The app is deployed **live on Render** for instant access.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Live Demo](#live-demo)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Future Scope](#future-scope)
- [Author](#author)

---

## Project Overview
This project predicts the likelihood of diabetes in patients based on medical features.  
It uses a **Random Forest Classifier** with **GridSearchCV** for hyperparameter optimization, and is deployed via **FastAPI**.  
The project is **Dockerized** and integrated with **CI/CD pipelines** for easy deployment.

---

## Dataset
The dataset contains patient health information with features like:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Target:** `Outcome` (0 = Non-diabetic, 1 = Diabetic)

---

## Technologies Used
- Python 
- Pandas & NumPy
- Scikit-learn (RandomForest, GridSearchCV, Pipelines)
- FastAPI (Backend)
- Docker (Containerization)
- CI/CD (GitHub Actions)
- Pickle (Model Serialization)

---

## Live Demo
Access the running application here:  https://diabhtml.onrender.com


---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mukulchaudhary1031/<repo-name>.git
cd <repo-name>

2. Create virtual environment & install dependencies:



python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt

3. Run FastAPI server locally:



uvicorn main:app --reload

4. (Optional) Run with Docker:



docker build -t diabetes-prediction .
docker run -p 8000:8000 diabetes-prediction


---

Usage

1. Open browser at:



http://127.0.0.1:8000

2. Or use the live Render link for immediate access.


3. Enter patient details in the form and submit.


4. Get prediction: diabetic or non-diabetic.


5. API testing is also possible via Postman or other HTTP clients.




---

Features

Preprocessing pipelines for numeric & categorical features

Handles missing data automatically

RandomForest with hyperparameter tuning (GridSearchCV)

Test accuracy & confusion matrix evaluation

FastAPI backend for REST API

Dockerized deployment with CI/CD ready

Live deployment available on Render



---

Future Scope

Add visualizations like feature importance

Integrate more ML models for comparison

Add user authentication & patient history tracking

Deploy on other cloud platforms (AWS/GCP)



---

Author

Mukul Chaudhary

GitHub: https://github.com/mukulchaudhary1031

FastAPI + Docker + CI/CD enthusiast

Passionate about AI/ML full-stack applications
