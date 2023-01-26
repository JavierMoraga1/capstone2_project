import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import tensorflow as tf

#Parameters
parser = argparse.ArgumentParser(description="Generate test predictions in a .csv file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-M","--model", help="model file")
parser.add_argument("-S","--size", help="image size")
parser.add_argument("-O","--output", help="output file")
args = parser.parse_args()
config = vars(args)

model_file = 'model.h5'
if config["model"] != None:
  model_file = config["model"]
img_size = 300
if config["size"] != None:
  img_size = config["size"]
output_file = 'predictions.csv'
if config["output"] != None:
  output_file = config["output"]

 # Loading and preparing the dataset
print ('Loading dataset and model...')
df_test = pd.read_csv('test.csv')
model = tf.keras.models.load_model(model_file)

# Doing the predictions
print("Making predictions with model %s..." % model_file)
healthy = []; multiple_diseases = []; rust = []; scab = []
for i in tqdm(df_test.index):
    img_path = './images/submit/' + df_test['image_id'][i] + '.jpg'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    x = np.array(img)
    X = np.array([x])
    X = tf.keras.applications.efficientnet_v2.preprocess_input(X)
    pred = model.predict(X, verbose=0)

    healthy.append(pred[0][0]); multiple_diseases.append(pred[0][1]); rust.append(pred[0][2]); scab.append(pred[0][3])

df_submit = df_test.loc[:, ['image_id']] # [:5]
df_submit['healthy'] = healthy; df_submit['multiple_diseases'] = multiple_diseases; df_submit['rust'] = rust; df_submit['scab'] = scab 

df_submit.to_csv(output_file,index=False)
print("File %s generated" % output_file)

