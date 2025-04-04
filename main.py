from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load("random_forest_model.pkl")
columns = joblib.load("feature_columns.pkl")

class AppFeatures(BaseModel):
    Rating: float
    Reviews: int
    Size: float
    Price: float
    Category: str
    Type: str
    Content_Rating: str
    Genres: str

@app.post("/predict")
def predict_app_success(data: AppFeatures):
    df_input = pd.DataFrame([[data.Rating, data.Reviews, data.Size * 1024, data.Price]], columns=["Rating", "Reviews", "Size", "Price"])
    # Dummy variables setup
    for col in columns:
        if col not in df_input.columns:
            df_input[col] = 0
    for col in ["Category", "Type", "Content Rating", "Genres"]:
        col_name = col.replace(" ", "_") + "_" + getattr(data, col.replace(" ", "_"))
        if col_name in columns:
            df_input[col_name] = 1
    df_input = df_input[columns]
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1] * 100
    result = "Popular" if prediction == 1 else "Unpopular"
    return {"result": result, "confidence": f"{prob:.2f}%"}

