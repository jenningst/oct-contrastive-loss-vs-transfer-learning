import os
import shutil
import random
from tkinter import CENTER


TRAIN_IMAGE_LIST_DIR = '../config'
TRAIN_IMAGE_LIST_FILE = 'training_image_list.txt'
LOCAL_IMG_DIR = '../../../../../large-project-data/OCT2017/train'
CORPUS_DIR = '../corpus'
SEED = 42
SAMPLE_SIZE = 25

CNV_FILES = set()
DME_FILES = set()
DRUSEN_FILES = set()
NORMAL_FILES = set()


def read_training_into_sets():
    for subdir in os.listdir(LOCAL_IMG_DIR):
        if subdir != '.DS_Store':
            subdir_files = os.listdir(os.path.join(LOCAL_IMG_DIR, subdir))
            if subdir == 'CNV':
                CNV_FILES.update(subdir_files)
            elif subdir == 'DME':
                DME_FILES.update(subdir_files)
            elif subdir == 'DRUSEN':
                DRUSEN_FILES.update(subdir_files)
            else:
                NORMAL_FILES.update(subdir_files)


def partition_training_images():
    with open(os.path.join(TRAIN_IMAGE_LIST_DIR, TRAIN_IMAGE_LIST_FILE), 'r') as infile:
        for i, line in enumerate(infile.readlines()):
            line = line.strip()
            prefix = line[:line.index('-')]
            if prefix == 'CNV':
                CNV_FILES.remove(line)
            elif prefix == 'DME':
                DME_FILES.remove(line)
            elif prefix == 'DRUSEN':
                DRUSEN_FILES.remove(line)
            else:
                NORMAL_FILES.remove(line)
    

def sample_classes(n: int=25):
    '''
    Samples n files from each class of the training dataset.
    '''

    random.seed(SEED)

    for subdir in os.listdir(LOCAL_IMG_DIR):
        print(f'Sampling {subdir}')
        # list the files and their indexes
        if subdir != '.DS_Store':
            subdir_files = os.listdir(os.path.join(LOCAL_IMG_DIR, subdir))

            if subdir == 'CNV':
                sample_indexes = random.sample([ i for i in range(len(CNV_FILES))], n)
            elif subdir == 'DME':
                sample_indexes = random.sample([ i for i in range(len(DME_FILES))], n)
            elif subdir == 'DRUSEN':
                sample_indexes = random.sample([ i for i in range(len(DRUSEN_FILES))], n)
            else:
                sample_indexes = random.sample([ i for i in range(len(NORMAL_FILES))], n)

            for s in sample_indexes:
                try:
                    shutil.copy(
                        src=os.path.join(LOCAL_IMG_DIR, subdir, subdir_files[s]),
                        dst=os.path.join(CORPUS_DIR, subdir_files[s])
                    )
                    print(f'Successfully sampled file {subdir_files[s]}')
                except Exception as err:
                    print(f'Error occurred during copy for file {subdir_files[s]}: {err}')
    print('Sampling completed')


if __name__ == "__main__":
    cnv_length_pre = len(CNV_FILES)
    read_training_into_sets()
    cnv_length_post = len(CNV_FILES)
    partition_training_images()
    assert cnv_length_post == cnv_length_pre + 5000, f'actuals {cnv_length_post}, {cnv_length_pre}'
    # sample_classes(n=25)
