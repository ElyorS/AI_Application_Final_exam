import os
import math
import string
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
from google.colab import drive
import click  # Importing click library for CLI

def num_to_label(num, alphabet):
    text = ""
    for ch in num:
        if ch == len(alphabet): # ctc blank
          break
        else:
          text += alphabet[ch]
    return text

def num_to_label(num, alphabet):
    text = ""
    for ch in num:
        if ch == len(alphabet): # ctc blank
          break
        else:
          text += alphabet[ch]
    return text


# Decode labels for softmax matrix

def decode_text(nums):
  values = get_value(
      ctc_decode(nums, input_length=np.ones(nums.shape[0])*nums.shape[1],
                 greedy=True)[0][0])

  texts = []
  for i in range(nums.shape[0]):
    value = values[i]
    texts.append(num_to_label(value[value >= 0], loaded_alphabet))
  return texts
def decode_text(nums):
  values = get_value(
      ctc_decode(nums, input_length=np.ones(nums.shape[0])*nums.shape[1],
                 greedy=True)[0][0])

  texts = []
  for i in range(nums.shape[0]):
    value = values[i]
    texts.append(num_to_label(value[value >= 0], alphabet))
  return texts

@click.group()
def cli():
    pass

@cli.command()
def mount_drive():
    drive.mount('/content/drive')

@cli.command()
def extract_files():
    !mkdir lines
    !mkdir ascii
    

@cli.command()
def load_labels():
   
@cli.command()
def preprocess_images():
    

@cli.command()
def train_model():
    

@cli.command()
def evaluate_model():
    

@cli.command()
def recognize_images(image_paths):
    
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        preprocessed_image = preprocess(image)
        image_data = np.expand_dims(preprocessed_image, axis=0)
        predicts = model.predict(image_data)
        predicts = decode_text(predicts)
        print(f"Image: {image_path}, Prediction: {predicts}")

if __name__ == '__main__':
    cli()
