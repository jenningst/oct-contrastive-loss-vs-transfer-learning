# Samsung Optical Coherence Tomography (OCT) Disease Prediction

Samsung (OCT) Disease Prediction is a utility that facilitates the diagnosis of retinal disease from optical coherence tomography images. This project utilizes supervised and semi-supervised techniques to generate models trained at multi-class disease classification and exposes model inference through a collection of REST-ful API endpoints. The full project is deployable as a stand-alone, containerized application or on popular cloud-computing environments.

## Background

The OCT (Optical Coherence Tomography) is an imaging method used to diagnose the patient’s retinal health into four categories: 

-  Normal  
-  CNV
-  DME
-  DRUSEN

### Goal

1. create a baseline: reproduce the SOTA classification accuracy by training a Deep CNN (Supervised Learning) We use Inception_V3 model for this part. 
2. improve the classification accuracy over the baseline in part 1 using the self-supervised method described in simclr paper.
	- Learn the Unsupervised Features based on the Self-Supervision technique 
	- Train a classification head on top of the learned Unsupervised Features to further improve the accuracy upon the established baseline. 

### Importance
* This tool can expedite the diagnosis of treatable diseases that can lead to blindness. 

* By doing so, medication can be prescribed in time for the patients and prevent them from becoming blind. 
 
* Additionally, the tool has the potential to be generalized in other applications of biomedical imaging including x-ray, MRI, and computer tomography)

To give some context, 30 million OCT scans are produced each year. Nearly 11 million people in the US alone will suffer from 1 of these diseases.

### Audience
* The outcome is a support tool that provides retinal specialists with accurate and timely diagnosis of key pathology in OCT images. 
* The OCT images can be used both for populations without proper access, and for those in overpopulated areas in order to reduce patient burden by using automated triage systems.

### Data source
[Mendeley] (https://data.mendeley.com/datasets/rscbjbr9sj/2)


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
- [Asieh Harati](https://github.com/AsiehH)
- [Wolfgang Black](https://github.com/wolfgangjblack)
- [Troy Jennings](https://github.com/jenningst)

## References
- [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867418301545%3Fshowall%3Dtrue)
- [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)
