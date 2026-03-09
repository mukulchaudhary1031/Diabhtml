import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.ensemble import RandomForestClassifier 


# --------------------------------
# 1) Load Dataset
# --------------------------------

df = pd.read_csv("C:\\Users\\chaud\\Downloads\\archive (1).zip")


print(df.head(9))


# --------------------------------
# 2) Split Features & Target
# --------------------------------

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

print(df.isnull().sum())


numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

all = numeric_cols + categorical_cols
print("calculate columns:", all)


# --------------------------------
# 5) Preprocessing Pipelines
# --------------------------------

numeric_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='mean')),
('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='most_frequent')),
('encoder', OneHotEncoder(handle_unknown='ignore'))
])



preprocessor = ColumnTransformer(
transformers=[
('num', numeric_transformer, numeric_cols),
('cat', categorical_transformer, categorical_cols)
]
)


# --------------------------------
# 6) Full Pipeline with SVM
# --------------------------------

pipeline = Pipeline(steps=[
('preprocessing', preprocessor),
('model', RandomForestClassifier())
])


# --------------------------------
# 7) GridSearch
# --------------------------------

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__bootstrap': [True, False]
}

grid = GridSearchCV(
pipeline,
param_grid=param_grid,
cv= 3,
scoring='accuracy'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


# --------------------------------
# 8) Test Accuracy
# --------------------------------

y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("confusion test:",confusion_matrix(y_test,y_pred))


# --------------------------------
# 9) Save Model
# --------------------------------

with open("model_Rb.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved successfully ✅")