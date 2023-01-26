import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

from typing import Optional    
from io import BytesIO
from urllib import request


types = [
    'healthy',
    'multiple_diseases',
    'rust',
    'scab'
]

# model_file = 'efficientnetv2b3_v1_1_48_0.878.h5'
model_file = 'model.h5'
model = tf.keras.models.load_model(model_file)

input_size = 300

class Img(BaseModel):
    url: Optional[str] = None
    class Config:
      schema_extra = {
        "example": {
          "url": "https://raw.githubusercontent.com/JavierMoraga1/capstone2_project/master/examples/Train_3.jpg"
        }            
      }

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


app = FastAPI()
@app.get("/ping")
def ping():
    return {"message": "PONG"}

@app.post("/predict")
def predict(image: Img):

    img = download_image(image.url)
    img = prepare_image(img,(input_size,input_size))
    
    x = np.array(img)
    X = np.array([x])
    X = tf.keras.applications.efficientnet_v2.preprocess_input(X)

    pred = model.predict(X)

    result = {
      'Healthy': float(pred[0][0]),
      'Multiple_diseases': float(pred[0][1]),
      'Rust': float(pred[0][2]),
      'Scab': float(pred[0][3])
    }

    return result