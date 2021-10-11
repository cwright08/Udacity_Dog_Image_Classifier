# Udacity Dog Image Classifier

Please note: An enhanced version of this read me is avaliable on Medium at the following address: 

Project Description: 
The aim of this project was to create a web application that takes a user provided image, runs it through a pre-trained alogirthm, and returns the dog breed (or the fact it is not a dog in the image!). I primarily chose to undertake this project as my Capstone project to conclude my Udacity Data Science nanodegree to 
further build my knowledge of neural networks, deep learning, and particularly CNNs which I had not encountered previously.  

The core problem the project attempts to solve was training a CNN to predict and classify dog breeds, a task that is notoriously difficult, as various dog breeds can be extremely hard to tell apart,
even to the trained human eye. This is described in detail in the notebook. The project successfully fulfilled this aim, achieveing accuracy of over 80% in testing data. In the real world however,
the accuracy and recall is likely to be lower. 

Web App - Instructions. 
The web app is not currently hosted, therefore you must run it on your local machine. Instructions are as follows:
1) Download all the source code under 'App' to your local PC. 
2) Ensure all the libraries and technologies listed below are installed and sufficiently up to date in your workspace. 
3) Navigate to the 'app.py' - run this in Python to start the app. 
4) In your browser (Chrome/Edge), navigate to http://localhost:3001/
5) Website is running. 

Acknowledgements:
The base notebook and structure of the project, along with data and a portion of the code was provided by Udacity (Udacity.com) as part of their Data Science Nanodegree. 

Methodology:
Data Exploration
As with all data science projects, the first step was the analyse the data that was avaliable to training. The data set contained 8,351 images of dogs in total. We also imported a dataset of 13,233 human images in order to detect a human face in the image. 

Human Face Detection
In order to detect human faces, we use OpenCVs(https://opencv.org/) Haar feature-based cascade classifier which is a pretrained human face detector. All that was neccessary was to download the XML for this model, and write the dog detector function. This function simply takes an input of a image, and returns True if it detects a human face in the image. I then tested the function on the human and dog images. The model returned a face in 100% of the human images, which was impressive. However it also returned a human face detection for 11% of dog images. 

Dog Detection
To detect the presence of a dog in an image, we utilise the pretrained Resnet-50 model from TensorFlow(https://www.tensorflow.org/). This is a model which has been trained on ImageNet, a database containing over 10 million labeled images. When given an image as input, the model returns a prediction of the category of object contained.

With the Resnet-50 model, it was first neccessary to pre-process the data to provide the Keras backend with a 4D array for prediction. There was also some additional pre-processing required to the image in order to prepare it for input to the model. Fortunatley, TensorFlow provides an implementation of this pre-processing step in its distribution. 

All that remained was to write the dog detection algorithm that returned True if a dog was detected in the supplied image. Again, the dog detector exhibited excellent performane, identifying 100% of the dog images and 0% of human images as dogs- this was very impressive for a pre-trained model. 

Dog Breed Detection
With the basic task of classifying an image as a dog or a human complete, it was time to tackle the harder problem of classifying dogs in to different breeds. In order to do this I first attempted to build a basic CNN from scratch using Keras. The model I chose contained 5 layers, 3 of which were convolutional and 2 were dense layers. 

Once compiled, I then trained the model. I attempted a number of configurations, with various number of training Epochs. The best configuration was achieved with two 0.25 drop off layers included, RMS prop optimizer, and 50 training epochs. This yielded an accuracy score of 10.7% - not bad for a first attempt, but not good enough for production.

To try and achieve greater levels of accuracy, I attempted to train a model using a transfer learning approach. Two different base alogirthms were tested, both VGG-19 and ResNet-50. ResNet 50 was found to show the better performance of the two models. 

The final step in the process was to write a function to take a user supplied image and return a dog breed or human face result.  

Results
While the ResNet-50 based model achieved a very good accuracy score using the testing dataset, it was neccessary to test it on completely unseen data. In order to do this, we sampled a number of user supplied dog and human images. The result of this was mixed. 


Web App 
In order to serve the dog breed predictor to the end user, I built a web application using Flask backend. The app ensured the user could easily upload their own image and return a result from the algorithm through an intuitive user experience. It also allowed me to further develop my front end development skills which was a bonus. 

In order to run the application, follow the instructions above. In future, this app could be deployed to the web for the most accessible and widely available user experience. 

Conclusion
To conclude, the project successully produced a web application which was capable of predicting whether a user provided image is a human or a dog, and if a dog the relevant breed. This was achieved using a Transfer Learning based approach with the ResNet-50 model as the base. Through completeing the project I have successully developed and demonstrated skills in CNNs and image classifcation. 

In future, the algorithm and models could undoubtabley be further tuned to provide greater levels of accuracy. One appraoch that could be used for this would be to use image augmentation. Further, larger quantities of training data could be used to better tune the model. 

Libraries and Technology
Python 3.8 Anaconda
Pandas
Numpy
Flask
Keras
glob
sklearn
random
cv2
matplotlib
tqdm
PIL



