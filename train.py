import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
import os
import shutil
import splitfolders

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


#Parameters
parser = argparse.ArgumentParser(description="Train model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-E","--nepochs", help="number of epochs")
parser.add_argument("-L","--lrate", help="learning rate")
parser.add_argument("-I","--isize", help="inner size")
parser.add_argument("-D","--drate", help="drop rate")
parser.add_argument("-O","--output", help="output file")
args = parser.parse_args()
config = vars(args)

input_size = 300
n_epochs = 50
if config["nepochs"] != None:
  n_epochs = int(config["nepochs"])
learning_rate = 0.001  
if config["lrate"] != None:
  learning_rate = float(config["lrate"])  
inner_size = 100  
if config["isize"] != None:
  inner_size = int(config["isize"])  
drop_rate = 0.0
if config["drate"] != None:
  drop_rate = float(config["drate"])  
output_file = 'model.h5'
if config["output"] != None:
  output_file = config["output"]

# Functions
def restructure_folders():
    # Loading datasets 
    train_df = pd.read_csv('train.csv')
    submit_df = pd.read_csv('test.csv')
    # Creating new folders
    train_dir = Path('./images/training')
    submit_dir = Path('./images/submit')
    for label in ['healthy', 'multiple_diseases', 'rust', 'scab']:
      d = train_dir / label
      d.mkdir(parents=True, exist_ok=True)  
    submit_dir.mkdir(parents=True, exist_ok=True)    
    # Moving the images
    count = 0
    for i in tqdm(train_df.index):
      img = train_df.image_id[i] + '.jpg'
      img_path = img_dir / img
      for label in ['healthy', 'multiple_diseases', 'rust', 'scab']:
          if train_df[label][i] == 1:
              new_path = train_dir.absolute() / label / img
              shutil.move(img_path, new_path)
              count += 1
    print(f'Total moved to images/training: {count}')
    count = 0
    for i in tqdm(submit_df.index):
      img = submit_df.image_id[i] + '.jpg'
      img_path = img_dir / img
      new_path = submit_dir / img
      shutil.move(img_path, new_path)
      count += 1
    print(f'Total moved to images/submit: {count}')
    # Splitting training images into 3 folders
    splitfolders.ratio(train_dir, output='./images/tmp', move=True, ratio=(0.8,0.1,0.1))
    shutil.rmtree(train_dir)
    os.rename('./images/tmp', train_dir)

def preprocess(input_size = 300):
    train_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
    )
    val_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input)
    test_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input)

    train_ds = train_gen.flow_from_directory(
        './images/training/train',
        target_size=(input_size, input_size),
        batch_size=32
    )
    val_ds = val_gen.flow_from_directory(
        './images/training/val',
        target_size=(input_size, input_size),
        batch_size=32
    )
    test_ds = test_gen.flow_from_directory(
        './images/training/test',
        target_size=(input_size, input_size),
        batch_size=32
    )
    return train_ds, val_ds, test_ds

def make_model(input_size=300, learning_rate=0.001, inner_size=100,
               drop_rate=0.5):

    base_model = keras.applications.EfficientNetV2B3(        
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))

    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    if inner_size > 0:
      inner = keras.layers.Dense(inner_size, activation='relu')(vectors)
      drop = keras.layers.Dropout(drop_rate)(inner)
    else:
      drop = keras.layers.Dropout(drop_rate)(vectors)
    
    outputs = keras.layers.Dense(4, activation='softmax')(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# Main
img_dir = Path('./images')
images = list(img_dir.glob('*.jpg'))
if len(images) > 0:
  print("Restructuring folder /images")
  restructure_folders()

print ('Loading and preparing the dataset...')
train_ds, val_ds, test_ds = preprocess(input_size)

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    inner_size=inner_size,
    drop_rate=drop_rate
)

print('Training model')
history = model.fit(train_ds, epochs=n_epochs, validation_data=val_ds,
                   callbacks=[checkpoint])

model_file = 'model.h5'
model = keras.models.load_model(model_file)

print('Evaluating with Validation dataset...')                   
eval = model.evaluate(val_ds)
print('Loss : ' + str(eval[0]))
print('Accuracy : ' + str(eval[1]))

print('Evaluating with Test dataset...')                   
eval = model.evaluate(test_ds)
print('Loss : ' + str(eval[0]))
print('Accuracy : ' + str(eval[1]))

print(f'The model is saved to {output_file}')
