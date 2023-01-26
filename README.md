# Plant Pathology 2020 - FGVC7

## Problem description
___
For my last capstone project I've choosen another dataset from a Kaggle competitition ([**Plant Pathology 2020 - FGVC7**](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)). It's a collection of images of apple tree leaves with various diseases. It is not a very large dataset, so you can easily train several models and test many configuration options, which was my main objective. 

The goal is to **classify a given image from the test dataset into 4 categories (leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease**). It is asked to calculate the probability of belong to each of the categories

To get a score in the competition, you must first built and train a model using a training dataset and then use another testing dataset to calculate a set of predictions. This competition is not currently active, but it is possible to submit the predictions to verify their accuracy.

### Used Datasets
___

Due to the large size of the dataset (almost 800 MB) it is not provided in the repository and must be downloaded from Kaggle

Description and dowload: https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data

- train.csv: Used to train the model. Variables:
  - image_id: the foreign key
  - healthy: one of the target labels
  - multiple_diseases: one of the target labels
  - rust: one of the target labels
  - scab: one of the target labels
- test.csv: Used to predict and submit the results
  - image_id: the foreign key
- /images: A folder containing the train and test images, in jpg format.


### Evaluation Metric
___

Submissions are evaluated on mean column-wise **ROC AUC**, i.e. the score is the average of the individual AUCs of each predicted column.


## Approach and solutions
___
- For this classification problem I've trained different types of algorithms of image classification and evaluated them using **accuracy**
- I've selectioned ___EfficientNetV2___ as the best model and tuned its parameters and several other options (image size, data augmentations, layers, etc)
- It is worth mentioning that, the differences between models have been very small. Given this small difference, I have chosen a fairly lightweight model (EfficientNetV2B3, around 51MB) for the deployment.

## Repo description
___
- `notebook.ipynb` -> Notebook for EDA, data cleaning, model testing and other preparatory steps
- `train.py` -> Script with the entire process to train the model from the initial loading of the dataset to the generation of the `.h5` file
- `predict.py` -> Creates an application that uses port 9696 and the endpoint `/predict` (`POST` method). The service:
  - Receive data request in json format
  - Validate and adapt the data to the model
  - Predict and return the result in json format
- `predict_test.py` -> Sends a test request and prints the result
- `generate_predictions.py` -> Script to generate the results to be sent to Kaggle. It creates the (`predictions.csv`) file
- `Pipfile`, `Pipfile.lock` -> For dependency management using Pipenv
- `Dockerfile` -> Required for build a docker image of the app with the 
predict service
- `model.h5` -> Model used by predict.py. A different file can be generated using train.py

## Dependencies Management
___
In order to manage the dependencies of the project you could:
- Install Pipenv => `pip install pipenv`
- Use it to install the dependencies => `pipenv install`

## Train the model
___
1. Enter the virtual environment => `pipenv shell`
2. Run script => `train.py`
or
1. Run directly inside the virtual environment => `pipenv run train.py`

Optional parameters:
- `-E --nepochs` (default 50)
- `-L --lrate` (default 0.001)
- `-I --isize` (default 100)
- `-D --drate` (default 0.0)
- `-O --output` (default `model.h5`)
## Run the app locally (on port 9696)
___
1. Enter the virtual environment => `pipenv shell`
2. Run script => `uvicorn predict:app --host 0.0.0.0 --port 9696`

## Using Docker
___
You can also build a Docker image using the provided `Dockerfile` file =>
`docker build . -t plantpathology`

To run this image locally => `docker run -it --rm -p 9696:9696 plantpathology`
## Deploying to AWS
___
Assuming you have a AWS account, you can deploy the image to AWS Elastic Beanstalk with the next steps:
- Install AWS Elastic Beanstalk CLI => `pip install awsebcli`
- Init EB configuration => `eb init -p docker -r eu-west-3 plantpathology` and authenticate with your credentials (If you want you can change eu-west-3 for your local zone)
- Create and deploy the app => `eb create plantpathology-env`

(**Until the end of the project review period the application will remain deployed on AWS Elastic Beanstalk** and accessible on plantpathology-env.eba-xabjczby.eu-west-3.elasticbeanstalk.com)

## Using the service
___
For a simple test request you have several options. You could:
1) Run the test script locally => `python predict_test.py` (needs `requests`. To install => `pip install requests`)

   Optional parameters:
    - ` -H --host` (`localhost:9696` by default)
    - ` -I --image` (`https://raw.githubusercontent.com/JavierMoraga1/capstone2_project/master/examples/Train_3.jpg` by default)
  
    For example, if you want test the AWS deployment with a different image you can run => `python predict_test.py --host plantpathology-env.eba-xabjczby.eu-west-3.elasticbeanstalk.com --image https://raw.githubusercontent.com/JavierMoraga1/capstone2_project/master/examples/Train_409.jpg`

    An example image is available for each category:
    - Train_3 (Rust)
    - Train_63 (Healthy)
    - Train_237 (Multiple diseases)
    - Train_409 (Scab)

2) Or use the browser for access the `/docs` endpoint => `localhost:9696/docs`  or `plantpathology-env.eba-xabjczby.eu-west-3.elasticbeanstalk.com/docs` and then use the UI to send a request (POST) to the `/predict` endpoint

3) Or use curl, Postman or another similar tool to send the request to the `/predict` endpoint

## Generating a file to submit the predictions to Kaggle
___
To generate a .csv file with the predictions => `pipenv run generate_predictions`. Optional parameters:
- `-M --model` (default `model.h5`)
- `-S --size` (default `300`)
- `-O --output` (default `predictions.csv`)












