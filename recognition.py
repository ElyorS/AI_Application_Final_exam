import cv2
import numpy as np
import pickle
import pytest
from recognition import preprocess, num_to_label, decode_text


@pytest.fixture
def sample_image():
    # Replace this with a path to a sample image in your dataset
    image_path = 'path/to/sample/image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return preprocess(image)

def test_preprocess(sample_image):
    # Ensure preprocess function works as expected
    processed_image = preprocess(sample_image)
    assert processed_image.shape == (64, 800)  # Adjust dimensions based on your preprocessing logic

def test_num_to_label():
    # Test num_to_label function
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.' "
    assert num_to_label([0, 1, 26, 27], alphabet) == "abAB"

def test_decode_text():
    # Test decode_text function
    nums = np.array([[0, 1, 2, 3], [26, 27, 28, 29]])
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.' "
    assert decode_text(nums, alphabet) == ["abcd", "efgh"]

