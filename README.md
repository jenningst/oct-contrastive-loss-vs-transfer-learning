# Samsung Optical Coherence Tomography (OCT) Disease Prediction

Samsung (OCT) Disease Prediction is a utility that facilitates the diagnosis of retinal disease from optical coherence tomography images. This project utilizes supervised and semi-supervised techniques to generate models trained at multi-class disease classification and exposes model inference through a collection of REST-ful API endpoints. The full project is deployable as a stand-alone, containerized application or on popular cloud-computing environments.

## Background

Coming Soon! 👀

## Getting Started

Model inference for supervised and semi-supervised learning methods are exposed through API endpoints using FastAPI. For convenience, the utility comes included with trained models -- an InceptionV3 neural network for supervised image classification and a ResNet-50 neural network utilizing the SimCLRv2 methodology -- available in the `src/classifier/model` directory. Furthermore, the repository includes a set of sample instances that were unseen from either model, all available in the `src/corpus` directory.

### Pre-Requisites
- Python 3.9+
- Docker 

## Using Samsung (OCT) Disease Prediction

### Building

The included `Dockerfile` allows for building a minimal container environment to run and test the API. 

To build a docker image from the project, run the following command from the root project directory:
```
docker build -t samsung-oct .
```

### Running
To create a new container, run the following command:
```
docker run -d --name <CONTAINER_NAME> samsung-oct
```

### Using the API
With the application running in a container, navigate to http://localhost:8000/ to use the basic "liveness" endpoint. Upon requesting the root ("/") endpoint, the API will respond with API status:

```
{
    "host": "localhost"
    "port": 8000
    "service": "inference"
    "status": "online"
    "docs": "http://localhost:8000/docs"
}
```

### Endpoints

Coming Soon! 👀

### Testing
Unit tests and API tests can be run after installing requirements:
#### Running Unit Tests
```
python -m pytest src/tests/unit/test_utilities.py
```

#### Running API Tests
```
python -m pytest src/tests/unit/test_utilities.py
```

## Contributors
- Asieh Harati
- Wolfgang Black
- [Troy Jennings](https://github.com/jenningst)

## References
- [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418301545%3Fshowall%3Dtrue)
- [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)
