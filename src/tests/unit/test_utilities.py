import os
import sys

sys.path.append(os.path.abspath("./utils"))

from utils import (
    check_for_valid_corpus,
    build_sampling_manifeset,
    create_sample_set
)

CORPUS_DIR = os.path.join(os.getcwd(), "corpus")

def test_corpus_is_valid():
    assert check_for_valid_corpus(corpus_path=CORPUS_DIR) == True, \
        f"Corpus is invalid"


def test_build_sampling_manifest_not_empty():
    manifest, instance_count, total_count = build_sampling_manifeset(corpus_path=CORPUS_DIR)
    assert instance_count > 0, \
        f"Expected instance count to be non-zero, got: {instance_count}"
    assert len(manifest) == 4, \
        f"Expected the number of classes to be 4, got: {len(manifest)}"
    assert total_count > 0, \
        f"Expected the total number of instances to be non-zero, got: {total_count}"


def test_stratified_sampling_one_returns_correct_samples():
    samples = create_sample_set(corpus_path=CORPUS_DIR, n=1, stratify=True)
    assert len(samples) == 4, \
        f"Expected stratified samples to be 4, got: {len(samples)}"


def test_stratified_sampling_returns_every_class():
    samples = create_sample_set(corpus_path=CORPUS_DIR, n=1, stratify=True)
    samples = [ s[:s.index("-")] for s in samples ]

    assert 'CNV' in samples, \
        f"Expected CNV sample to be in the list, got: {samples}"
    assert 'DME' in samples, \
        f"Expected DME sample to be in the list, got: {samples}"
    assert 'DRUSEN' in samples, \
        f"Expected DRUSEN sample to be in the list, got: {samples}"
    assert 'NORMAL' in samples, \
        f"Expected NORMAL sample to be in the list, got: {samples}"


def test_stratified_sampling_greater_than_one_returns_correct_samples_per_class():
    samples = create_sample_set(corpus_path=CORPUS_DIR, n=2, stratify=True)
    assert len(samples) == 8, \
        f"Expected stratified samples to be 8, got: {len(samples)}"


def test_stratified_sampling_greater_than_class_counts_returns_max_samples_per_class():
    samples = create_sample_set(corpus_path=CORPUS_DIR, n=6, stratify=True)
    assert len(samples) == 20, \
        f"Expected stratified samples to be 4, got: {len(samples)}"


def test_sampling_one_returns_single_sample():
    samples = create_sample_set(corpus_path=CORPUS_DIR, n=1, stratify=False)
    assert len(samples) == 1, \
        f"Expected stratified samples to be 1, got: {len(samples)}"


def test_sampling_greater_than_one_returns_correct_samples():
    samples = create_sample_set(corpus_path=CORPUS_DIR, n=10, stratify=False)
    assert len(samples) == 10, \
        f"Expected stratified samples to be 10, got: {len(samples)}"


def test_sampling_greater_than_n_returns_max_samples():
    samples = create_sample_set(corpus_path=CORPUS_DIR, n=1000, stratify=False)
    assert len(samples) == 20, \
        f"Expected stratified samples to be 20, got: {len(samples)}"


def test_stratified_sample_is_deterministic():
    samples1 = create_sample_set(corpus_path=CORPUS_DIR, n=1, stratify=True)
    samples2= create_sample_set(corpus_path=CORPUS_DIR, n=1, stratify=True)
    assert samples1 == samples2, \
        f"Expected sampling to be the same between runs, got sample set 1: {samples1} and sample set 2: {samples2}"