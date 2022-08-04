import itertools
import os
import random
from typing import List


def check_for_valid_corpus(corpus_path:str = None) -> bool:
    """
    Validates that the included corpus is not empty and has the same number
    of instances for each of the 4 classes.
    """
    
    subdirs = os.listdir(corpus_path)
    if ".DS_Store" in subdirs:
        subdirs.remove('.DS_Store') # remove hidden files for mac

    is_valid = False
    # check if the length of the size of the corpus is non-zero
    if subdirs:
        # check that each subdir is not empty and contains the same number of samples
        image_count = 0
        for i, dir in enumerate(subdirs):
            # check if the current directory is empty
            directory_image_count = len(os.listdir(os.path.join(corpus_path, dir)))
            if directory_image_count < 1:
                print(f"Directory {dir} contains no instances")
                break
            else:
                if i == 0:
                    # if the first subdirectory, update the image counter and check the next directory
                    image_count += directory_image_count
                else:
                    if directory_image_count != image_count:
                        print(f"Directory {dir} contained {directory_image_count} samples; expected {image_count}")
                        break
                    else:
                        # update the image count and check the next directory
                        image_count = directory_image_count
            is_valid = True
    else:
        print("Corpus contains no class (label) directories")

    return is_valid


def build_sampling_manifeset(corpus_path:str = None) -> List[List[str]]:
    """
    Builds a manifest of the included corpus as a list for sampling.
    """

    # build a manifest of the corpus
    manifest = [ None for i in range(4) ] # build the master manifest for the 4 classes
    subdirs = os.listdir(corpus_path)
    if ".DS_Store" in subdirs:
        subdirs.remove('.DS_Store') # remove hidden files for mac

    # update the manifest with the images from each class directory
    for i, dir in enumerate(subdirs):
        class_count = len(os.listdir(os.path.join(corpus_path, dir)))
        manifest[i] = os.listdir(os.path.join(corpus_path, dir))
    # get the total count of instances
    total_count = sum([ len(m) for m in manifest ])

    return (manifest, class_count, total_count)


def create_sample_set(corpus_path:str, n:int = 1, stratify:bool = False, random_state:int = 42) -> List:
    """
    Samples the corpus manifest in a stratified or non-stratified approach and returns a list of instances.
    """

    random.seed(random_state)

    # build the manifest and sample cache
    manifest, class_count, total_count = build_sampling_manifeset(corpus_path)
    class_samples = []

    if stratify:
        # if the number of instances in each class is less than n, just return n samples
        if n > class_count:
            print("n is less than the available instances in the class, returning 1 sample per class")
            n = class_count
        # sample n instances from each class label directory
        for class_index in range(len(manifest)):
            class_samples.extend(random.sample(manifest[class_index], int(n)))
    else:
        if n > total_count:
            print("n is less than the available instances in the corpus, returning max samples")
            n = total_count
        # create a flattened list and shuffle it, then sample n instances from all available instances
        flat_manifest = list(itertools.chain(*manifest))
        random.shuffle(flat_manifest)
        class_samples.extend(random.sample(flat_manifest, int(n)))

    return class_samples
    