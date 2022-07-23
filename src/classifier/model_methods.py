import numpy as np
import os
import random
import tensorflow as tf

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
CORPUS_PATH = './corpus'
MODEL_PATH = './model/saved/simclr_net'

# suppress tensorflow output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# load model
print('Loading Tensorflow SavedModel()')
model = tf.keras.models.load_model(MODEL_PATH)


def predict_from_param():
    pass


def predict_from_corpus():
    # return a model prediction and supporting info for an instance from the corpus

    # get a random image
    images = os.listdir(CORPUS_PATH)

    if len(images) < 1:
        response = {
            'error': 'No images in the corpus'
        }
    else:
        # get a random instance
        index = random.randrange(0, len(images))
        image_name = images[index]

        # preprocess the image and resize
        img = tf.keras.preprocessing.image.load_img(
            path=os.path.join(CORPUS_PATH, images[index]),
            target_size=IMAGE_SIZE
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img=img)
        img_array = tf.expand_dims(img_array, 0)

        # make predictions and return label and prediction
        pred = model.predict(img_array)
        response = { 
            'instance': image_name,
            'label': image_name[:image_name.index('-')],
            'prediction': CLASS_NAMES[np.argmax(pred[0])] 
        }
    return response