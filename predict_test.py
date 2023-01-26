import requests
import argparse
 
parser = argparse.ArgumentParser(description="Try predict service",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-H","--host", help="host of the service")
args = parser.parse_args()
config = vars(args)

if config["host"] == None:
  host = 'localhost:9696'
else:
  host = config["host"]

url = f'http://{host}/predict'

image = {
  "url": "https://raw.githubusercontent.com/JavierMoraga1/capstone2_project/master/examples/Train_3.jpg",
}

print('Image: ', image)

response = requests.post(url, json=image).json()
print('Prediction: ', response)