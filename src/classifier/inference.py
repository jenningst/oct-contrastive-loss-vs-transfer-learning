import json
import os

import uvicorn

from enum import Enum
from typing import Union

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_methods import (
    predict_from_corpus, 
    predict_from_param,
    get_classification_report_from_corpus,
    get_liveness
)


load_dotenv()

# setup api w/ CORS
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# model class for handling predictions using different models
class ModelName(str, Enum):
    inception = 'inceptionv3'
    simclr = 'simclr_student'

class LivenessOut(BaseModel):
    host: str
    port: int
    service: str
    status: str
    docs: str

# # TODO: finish classification models, if feasible
# class ClassificationScore(BaseModel):
#     precision: float
#     recall: float
#     f1_score: float # TODO: update the classification report object to include f1_score instead of f1-score key
#     support: int

# class ClassificationOut(BaseModel):
#     report: Union[ClassificationScore, None] == None



@api.get('/', response_model=LivenessOut)
async def root():
    # endpoint for api liveness status
    response = get_liveness()
    return response


@api.get('classification_report/')
async def classification_report(model_name: ModelName):
    # endpoint to get classification report for entire sample (batteries-incl.) corpus
    response = get_classification_report_from_corpus(model_name.value)
    return response


# # TODO: this endpoint needs to stream in image bytes as a payload
# @api.get('/predict/')
# async def predict(image: UploadFile):
#     # endpoint for inference on parameterized instance
#     response = predict_from_param(image)
#     return json.dumps(response)


# NOTE: This endpoint gets removed after the /predict endpoint is working
# @api.get('/corpus_predict/{num_samples}')
# async def predict(model_name: ModelName, num_samples: int=1):
#     # endpoint to do inference on instance from curated corpus
#     response = predict_from_corpus(model_name.value, num_samples)
#     return response


if __name__ == '__main__':
    # uvicorn.run(api, host='127.0.0.1', port=8000)

    # TODO: remove the code below after finalization
    # resp = predict_from_corpus(20)
    # print(resp)

    print(get_classification_report_from_corpus(model_name='simclr_student'))