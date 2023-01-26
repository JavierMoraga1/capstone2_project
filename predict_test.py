import requests
import argparse
 
parser = argparse.ArgumentParser(description="Try predict service",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-H","--host", help="host of the service")
parser.add_argument("-I","--image", help="url of the image")
args = parser.parse_args()
config = vars(args)

host = 'localhost:9696'
if config["host"] != None:
  host = config["host"]
image = "https://raw.githubusercontent.com/JavierMoraga1/capstone2_project/master/examples/Train_3.jpg"
if config["image"] != None:
  image = config["image"]

url_service = f'http://{host}/predict'
url_image = {
  "url": image
}

print('Image: ', image)
response = requests.post(url_service, json=url_image).json()
print('Prediction: ', response)