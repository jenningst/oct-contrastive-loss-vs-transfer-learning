<p align='center'>
  <img width="600" src='https://github.com/jenningst/oct-contrastive-loss-vs-transfer-learning/blob/main/assets/amanda-dalbjorn-UbJMy92p8wk-unsplash.jpg' alt='Woman Eye'>
</p>
<p align='center'>
  Photo by <a href="https://unsplash.com/@amandadalbjorn?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Amanda Dalbjörn</a> on <a href="https://unsplash.com/s/photos/eye?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
</p>

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
docker run -d --name <CONTAINER_NAME> -p 8000:8000 samsung-oct
```

**NOTE** The API may not be immediately available after running a container since the saved models need to be loaded.

## Endpoints
### Liveness Test
The root endpoint to test whether the API service is running and functional. 

#### Request
```http
GET /
```

#### Parameters
<!-- | Parameter | Type | Description |
| :--- | :--- | :--- |
| `api_key` | `string` | **Required**. Your Gophish API key | -->
`None`

#### Response
```
{
    "service": "inference"
    "status": "online"
}
```

### Classification Report
An endpoint to return a classification report from all instances in the corpus. Either the `inceptionv3` or `simclrv2` models can be used.

#### Request
```http
GET /classification_report/?model_name={{model_name}}
```

#### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model_name` | `string` | **Required**. Either `inceptionv3` or `simclr` to generate a classification report. |


#### Response
```
{
    "classification_report": {
        "0": {
            "precision": 0.6666666666666666,
            "recall": 1.0,
            "f1-score": 0.8,
            "support": 4
        },
        "1": {
            "precision": 0.375,
            "recall": 0.75,
            "f1-score": 0.5,
            "support": 4
        },
        "2": {
            "precision": 1.0,
            "recall": 0.5,
            "f1-score": 0.6666666666666666,
            "support": 4
        },
        "3": {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": 4
        },
        "accuracy": 0.5625,
        "macro avg": {
            "precision": 0.5104166666666666,
            "recall": 0.5625,
            "f1-score": 0.4916666666666667,
            "support": 16
        },
        "weighted avg": {
            "precision": 0.5104166666666666,
            "recall": 0.5625,
            "f1-score": 0.4916666666666667,
            "support": 16
        }
    }
}
```

### Corpus Predict
An endpoint to return predictions for varying number of instances from the corpus.

**NOTE**: When stratification is used, sampling across each class will be conducted. The API will return the max number of instances in teach class if `num_samples` exceeds the total available instances in a given class in the corpus. When stratification is not used, random sampling across classes will be conducted.

#### Request
```http
GET /corpus_predict/?model_name={{model_name}}&num_samples={{num_samples}}&stratify={{stratify}}
```

#### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model_name` | `string` | **Required**. Either `inceptionv3` or `simclr` for running inference. Defaults to `inceptionv3`. |
| `num_samples` | `int` | The number of samples to run inference against. Defaults to 1. |
| `stratify` | `bool` | Indicator for whether sampling should be stratified across all classes. Defaults to `False`. |


#### Response
```
{
    "predictions": [
        {
            "index": 0,
            "instance": "DRUSEN-224974-4.jpeg",
            "label": "DRUSEN",
            "prediction": 0
        }
    ]
}
```

## Testing
Unit tests and API tests can be run after installing requirements:
### Running Unit Tests
```
python -m pytest src/tests/unit/test_utilities.py
```

### Running API Tests
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
