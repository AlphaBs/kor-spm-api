from fastapi import APIRouter, Depends, status, HTTPException
from pydantic import BaseModel
from sqlalchemy import Connection
from tf_keras.preprocessing.sequence import pad_sequences
from tokenizers import Tokenizer
import pandas as pd
from cleanutils import clean_message
import requests
import json
import numpy as np
import os
from services.database import get_db
from services.api_keys import get_api_key_by_key
from fastapi.security import APIKeyHeader
from limiter import limiter

router = APIRouter()

TF_SERVING_URL = os.environ.get('TF_SERVING_URL', 'http://localhost:8501/v1/models/cnn-spm:predict')
tokenizer = Tokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(db: Connection = Depends(get_db), api_key: str = Depends(api_key_header)):
    key = get_api_key_by_key(db, api_key)
    if key:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )

def tokenize_test(X_test):
    max_len = 200
    X_test = X_test.apply(lambda x: tokenizer.encode(x).ids)
    X_test = pad_sequences(X_test, maxlen=max_len)
    return X_test

class PredictRequest(BaseModel):
    message: str

class PredictResponse(BaseModel):
    spam_probability: float

@router.post("/model/predict", tags=["model"], dependencies=[limiter])
def predict(request: PredictRequest, api_key: str = Depends(verify_api_key)) -> PredictResponse:
    cleaned_message = clean_message(request.message, False)
    X_test = pd.Series([cleaned_message])
    X_test_tokenized = tokenize_test(X_test)
    
    data = json.dumps({
        "signature_name": "serving_default",
        "instances": X_test_tokenized.tolist()
    })
    
    headers = {"content-type": "application/json"}
    response = requests.post(
        TF_SERVING_URL,
        data=data,
        headers=headers
    )
    
    result = json.loads(response.text)
    prediction = np.array(result['predictions'])
    probability = prediction[0][0] * 100
    return PredictResponse(spam_probability=float(probability))
