from tkinter import image_names
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.metrics import classification_report

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
# CLASS_NAMES = ['CNV', 'NORMAL', 'DRUSEN', 'DME'] # NOTE: THIS WAS ASIEH'S IDEA!
CORPUS_PATH = '../corpus'
MODEL_PATH = '../model/saved/simclr_net'
SEED = 42

# suppress tensorflow output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# load model
print('Loading Tensorflow SavedModel()')
model = tf.keras.models.load_model(MODEL_PATH)


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

