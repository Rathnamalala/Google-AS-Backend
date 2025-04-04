from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model and feature columns
model = joblib.load("random_forest_model.pkl")
columns = joblib.load("feature_columns.pkl")

# Allow CORS (Cross-Origin Resource Sharing) to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (including OPTIONS, POST)
    allow_headers=["*"],  # Allows all headers
)

# Define the Pydantic model for input validation
class AppFeatures(BaseModel):
    Rating: float
    Reviews: int
    Size: float
    Price: float
    Category: str
    Type: str
    Content_Rating: str
    Genres: str

# Define the prediction endpoint
@app.post("/predict")
def predict_app_success(data: AppFeatures):
    # Create the DataFrame with input data
    df_input = pd.DataFrame([[data.Rating, data.Reviews, data.Size * 1024, data.Price]], columns=["Rating", "Reviews", "Size", "Price"])

    # Add dummy variables for missing columns based on feature_columns.pkl
    for col in columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Process categorical columns (one-hot encoding)
    for col in ["Category", "Type", "Content_Rating", "Genres"]:
        col_name = col.replace(" ", "_") + "_" + getattr(data, col.replace(" ", "_"))
        if col_name in columns:
            df_input[col_name] = 1

    # Ensure the input DataFrame has all the columns in the model's expected order
    df_input = df_input[columns]

    # Make prediction using the trained model
    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1] * 100  # Probability of being "Popular"

    # Determine the result
    result = "Popular" if prediction == 1 else "Unpopular"
    
    # Return the prediction result and confidence level
    return {"result": result, "confidence": f"{prob:.2f}%"}

