from fastapi import FastAPI
from joblib import load
import numpy as np
import pandas as pd

app = FastAPI()

model = load('compressed_pipeline.pkl')

@app.post("/predict")
async def predict(input: str):
    labels = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    prediction = model.predict([input]).tolist()
    prediction = prediction[0]
    results = [label for i, label in enumerate(labels) if prediction[i]]
    if len(results) == 0:
        results = "No toxic or offensive content detected"
    return {"results": results}
