import os


def check_for_valid_corpus(corpus_path: str):
    # check that the included corpus is not empty
    subdirs = os.listdir(corpus_path)
    subdirs.remove('.DS_Store') # remove hidden files for mac

    is_valid = False
    if subdirs:
        if sum([ len(os.listdir(os.path.join(corpus_path, d))) for d in subdirs ]) > 0:
            is_valid = True

    return is_valid