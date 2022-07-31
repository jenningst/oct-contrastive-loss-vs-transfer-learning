import os
import random
import sys

import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.getcwd()))
from utils.utils import check_for_valid_corpus

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
CORPUS_PATH = '../corpus'
SIMCLR_MODEL_PATH = '../model/saved/simclr_net'
INCEPTION_MODEL_PATH = '../model/saved/inceptionv3_net'
SEED = 42

load_dotenv()

# suppress tensorflow output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load model
print('Loading Tensorflow saved models ...')
simclr_model = tf.keras.models.load_model(SIMCLR_MODEL_PATH)
# inception_model = tf.keras.models.load_model(INCEPTION_MODEL_PATH)

def get_liveness():
    # return basic api liveness information
    return { 
        'host': os.getenv("API_HOST"),
        'port': int(os.getenv("API_PORT")),
        'service': 'inference',
        'status': 'online',
        'docs': f"http://{os.getenv('API_HOST')}:{os.getenv('API_PORT')}/docs"
    }


def get_classification_report_from_corpus(model_name: str):
    # endpoint to do classification for the batteries-included corpus

    if model_name == 'inceptionv3':
        # model = inception_model
        pass
    else:
        model = simclr_model
    
    if not check_for_valid_corpus(CORPUS_PATH):
        response = {
            "error": "Corpus has no directories or is emtpy."
        }
    else:
        ground_truth_labels = []

        # collect the ground-truth label indexes from the corpus
        subdirs = os.listdir(CORPUS_PATH)
        subdirs.remove('.DS_Store')

        for subdir in subdirs:
            image_list = os.listdir(os.path.join(CORPUS_PATH, subdir))
            for image in image_list:
                ground_truth_labels.append(CLASS_NAMES.index(image[:image.index('-')]))

        # load the dataset and make predictions to get the predicted class label indexes
        test_data = tf.keras.utils.image_dataset_from_directory(
            directory=os.path.join(CORPUS_PATH),
            labels='inferred',
            label_mode='categorical',
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            shuffle=True,
            seed=SEED,
        )

        y_pred = model.predict(test_data)
        predictions = [ CLASS_NAMES.index(CLASS_NAMES[np.argmax(p)]) for p in y_pred ]

        # build classification report
        report = classification_report(
            y_true=ground_truth_labels, 
            y_pred=predictions, 
            output_dict=True,
            zero_division=0
        )

        response = {
            'classification_report': report
        }
    
    return response


def predict_from_param():
    pass


def predict_from_corpus(n:int = 1, stratify:bool = False):

    # caches for the ground truth and predictions
    ground_truth = []
    y_pred = []
    subdirs = os.listdir(CORPUS_PATH)
    subdirs.remove('.DS_Store')

    for subdir in subdirs:
        image_list = os.listdir(os.path.join(CORPUS_PATH, subdir))
        for image in image_list:
            # append to ground truth for classification report
            ground_truth.append(CLASS_NAMES.index(image[:image.index('-')]))

    # TODO: if we can use this and still maintain perspective to the 
    # image and its correct class, then we should use this util
    # load validation (test) data
    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=os.path.join(CORPUS_PATH),
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
    )

    print(ground_truth)
    y_pred = model.predict(test_data)
    print(y_pred)
    pred = [ CLASS_NAMES[np.argmax(p)] for p in y_pred ]
    print(pred)

    # # do classification report
    # report = classification_report(
    #     y_true=ground_truth, 
    #     y_pred=y_pred, 
    #     output_dict=True,
    #     zero_division=0)

    # response = { 
    #     # 'instance': imge_name,
    #     # 'label': image_name[:image_name.index('-')],
    #     # 'prediction': CLASS_NAMES[np.argmax(pred[0])],
    #     'report': report
    # }
    # return response


# def predict_from_corpus(n:int = 1, stratify:bool = False):

#     # caches for the ground truth and predictions
#     ground_truth = []
#     y_pred = []
#     image_list = os.listdir(CORPUS_PATH)
#     image_list.remove('.DS_Store')

#     if len(image_list) < 1:
#         response = {
#             'error': 'No images in the corpus'
#         }
#     else:
#         # get n random instances from the corpus
#         # TODO: if we choose to stratify ,this code changes and the corpus needs to have subdirs
#         print(type(len(image_list)))
#         print(type(n))
#         indices = random.sample([ i for i in range(len(image_list))], int(n))
#         image_names = [image_list[i] for i in indices]

#         for image in image_names:
#             # print(image)
#             # append to ground truth for classification report
#             ground_truth.append(CLASS_NAMES.index(image[:image.index('-')]))

#             # preprocess the image and resize
#             img = tf.keras.preprocessing.image.load_img(
#                 path=os.path.join(CORPUS_PATH, image),
#                 target_size=IMAGE_SIZE
#             )
#             img_array = tf.keras.preprocessing.image.img_to_array(img=img)
#             img_array = tf.expand_dims(img_array, 0)

#             # make predictions and return label and prediction
#             pred = model.predict(img_array)
#             y_pred.append(CLASS_NAMES.index(CLASS_NAMES[np.argmax(pred[0])]))

#         # # TODO: if we can use this and still maintain perspective to the 
#         # # image and its correct class, then we should use this util
#         # # load validation (test) data
#         # test_data = tf.keras.utils.image_dataset_from_directory(
#         #     directory=os.path.join(CORPUS_PATH),
#         #     labels='inferred',
#         #     label_mode='categorical',
#         #     batch_size=BATCH_SIZE,
#         #     image_size=IMAGE_SIZE,
#         #     shuffle=True,
#         #     seed=SEED,
#         # )

#         # print(ground_truth)
#         # print(y_pred)

#         # do classification report
#         report = classification_report(
#             y_true=ground_truth, 
#             y_pred=y_pred, 
#             output_dict=True,
#             zero_division=0)

#         response = { 
#             # 'instance': imge_name,
#             # 'label': image_name[:image_name.index('-')],
#             # 'prediction': CLASS_NAMES[np.argmax(pred[0])],
#             'report': report
#         }
#     return response




# # SAVING FOR LATER; TODO: REMOVE AFTER WE DETERMINE WE DON'T NEED THIS
# def predict_from_corpus():
#     # return a model prediction and supporting info for an instance from the corpus

#     # get a random image
#     images = os.listdir(CORPUS_PATH)

#     if len(images) < 1:
#         response = {
#             'error': 'No images in the corpus'
#         }
#     else:
#         # get a random instance
#         index = random.randrange(0, len(images))
#         image_name = images[index]

#         # preprocess the image and resize
#         img = tf.keras.preprocessing.image.load_img(
#             path=os.path.join(CORPUS_PATH, images[index]),
#             target_size=IMAGE_SIZE
#         )
#         img_array = tf.keras.preprocessing.image.img_to_array(img=img)
#         img_array = tf.expand_dims(img_array, 0)

#         # make predictions and return label and prediction
#         pred = model.predict(img_array)
#         response = { 
#             'instance': image_name,
#             'label': image_name[:image_name.index('-')],
#             'prediction': CLASS_NAMES[np.argmax(pred[0])] 
#         }
#     return response

