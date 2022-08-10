import os
import random
import sys

import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from sklearn.metrics import classification_report

from utils import utils

IMAGE_SIZE_SIMCLR = (224, 224)
IMAGE_SIZE_INCEPTION = (299, 299)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
CORPUS_PATH = './corpus'
SIMCLR_MODEL_PATH = './model/saved/simclrv2_net'
INCEPTION_MODEL_PATH = './model/saved/inceptionv3_net'
SEED = 42

load_dotenv()

# suppress tensorflow output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load model
print('Loading Tensorflow saved models ...')
simclr_model = tf.keras.models.load_model(SIMCLR_MODEL_PATH)
inception_model = tf.keras.models.load_model(INCEPTION_MODEL_PATH)

def get_liveness():
    """
    # Endpoint to return basic api liveness information.

    :return: dictionary of basic api information/statistics
    :rtype: dict
    """
    return { 
        'service': 'inference',
        'status': 'online',
     }


def get_classification_report_from_corpus(model_name:str = "simclrv2"):
    """
    Endpoint to return a classification report for all sample instances in the included corpus.

    :param model_name: name of the model for inference, defaults to None
    :type model_name: str, optional
    :return: classification report as a json (dict) object
    :rtype: dict
    """

    if model_name == "inceptionv3":
        model = inception_model
        image_size = IMAGE_SIZE_INCEPTION
    else:
        model = simclr_model
        image_size = IMAGE_SIZE_SIMCLR
    
    # check for invalid directory contents
    if not utils.check_for_valid_corpus(CORPUS_PATH):
        response["error"] = "Corpus has no directories or is empty."
    else:
        ground_truth_labels = []
        images = []

        # collect the ground-truth label indexes from the corpus
        subdirs = os.listdir(CORPUS_PATH)
        if ".DS_Store" in subdirs:
            subdirs.remove(".DS_Store") # remove hidden files for mac

        # create a list of the ground-truth labels
        for subdir in subdirs:
            image_list = os.listdir(os.path.join(CORPUS_PATH, subdir))
            images.extend(image_list)
            for image in image_list:
                if image != ".DS_Store":
                    # add the class label index and images to cache
                    ground_truth_labels.append(CLASS_NAMES.index(image[:image.index("-")]))

        # make prediction on each sample image
        y_pred = []
        for idx, image in enumerate(images):
            # preprocess the image and resize
            img = tf.keras.preprocessing.image.load_img(
                path=os.path.join(CORPUS_PATH, image[:image.index("-")], image),
                target_size=image_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img=img)
            img_array = tf.expand_dims(img_array, 0)

            # make predictions and add to response
            preds = model.predict(img_array)
            y_pred.append(np.argmax(preds[0]))

        # build classification report
        report = classification_report(
            y_true=ground_truth_labels, 
            y_pred=y_pred, 
            output_dict=True,
            zero_division=0
        )

        response = {
            "classification_report": report
        }
    
    return response


def predict_from_corpus(model_name:str = "simclrv2", n:int = 1, stratify:bool = False):
    """
    Endpoint to run inference for n instances from the corpus.

    :param n: number of samples to run inference on, defaults to 1
    :type n: int, optional
    :param stratify: indicator to stratify the sampling or not, defaults to False
    :type stratify: bool, optional
    :param model_name: name of the model for inference, defaults to "simclr_model"
    :type model_name: str, optional
    :param random_state: random seed for reproducibility, defaults to 42
    :type random_state: int, optional
    :return: 
    :rtype: _type_
    """

    if model_name == "inceptionv3":
        model = inception_model
        image_size = IMAGE_SIZE_INCEPTION
    else:
        model = simclr_model
        image_size = IMAGE_SIZE_SIMCLR
    
    response = {}

    # check for invalid directory contents
    if not utils.check_for_valid_corpus(CORPUS_PATH):
        response["error"] = "Corpus has no directories or is empty."
    else:
        # sample the corpus
        class_samples = utils.create_sample_set(
            corpus_path=CORPUS_PATH,
            n=n, 
            stratify=stratify,
        )

        prediction_responses = []
        # make prediction on each sample image
        for idx, image in enumerate(class_samples):
            # preprocess the image and resize
            img = tf.keras.preprocessing.image.load_img(
                path=os.path.join(CORPUS_PATH, image[:image.index("-")], image),
                target_size=image_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img=img)
            img_array = tf.expand_dims(img_array, 0)

            # make predictions and add to response
            pred = model.evaluate(img_array)
            instance_prediction_response = {
                "index": idx,
                "instance": image,
                "label": image[:image.index("-")],
                "prediction": CLASS_NAMES.index(CLASS_NAMES[np.argmax(pred[0])])
            }

            prediction_responses.append(instance_prediction_response)
        # collect all responses
        response["predictions"] = prediction_responses

        return response


# def predict_from_image_upload(model_name:str = "simclr_model"):
#     pass