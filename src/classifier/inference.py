import json
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from classifier.model_methods import predict_from_corpus, predict_from_param

# setup api w/ CORS
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get('/')
async def root():
    # endpoint for basic liveness test
    return { 
        'service': 'inference',
        'status': 'online',
    }


@api.get('/predict/{image}')
async def predict(image):
    # endpoint for inference on parameterized instance
    response = predict_from_param(image)
    return json.dumps(response)


@api.get('/corpus_predict/')
async def predict():
    # endpoint for inference on instance from curated corpus
    response = predict_from_corpus()
    return response


if __name__ == '__main__':
    uvicorn.run(api, host='0.0.0.0', port=8000)