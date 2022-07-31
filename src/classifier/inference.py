import json
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_methods import predict_from_corpus, predict_from_param

# setup api w/ CORS
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # TODO: response models; e.g...
# class Item(BaseModel):
#     name: str
#     description: Union[str, None] = None
#     price: float
#     tax: Union[float, None] = None
#     tags: List[str] = []



@api.get('/')
async def root():
    # endpoint for basic liveness test
    return { 
        'service': 'inference',
        'status': 'online',
    }


# TODO: this endpoint needs to stream in image bytes as a payload
@api.get('/predict/{image}')
async def predict(image):
    # endpoint for inference on parameterized instance
    response = predict_from_param(image)
    return json.dumps(response)


@api.get('/corpus_predict/{num_samples}')
async def predict(num_samples):
    # endpoint for inference on instance from curated corpus
    response = predict_from_corpus(num_samples)
    return response


if __name__ == '__main__':
    # uvicorn.run(api, host='127.0.0.1', port=8000)

    # TODO: remove the code below after finalization
    resp = predict_from_corpus(20)
    print(resp)