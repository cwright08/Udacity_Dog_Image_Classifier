#Predictions
from resources import app
from keras.models import load_model 
from keras.preprocessing import image                  
import numpy as np
from tqdm import tqdm
from PIL import ImageFile                            
import pickle
import cv2                
import matplotlib.pyplot as plt   
ImageFile.LOAD_TRUNCATED_IMAGES = True  
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

def dog_algo (img_path):
    '''Function to allow image path to be passed and algorithm run and returned'''

    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    # load list of dog names
    dog_list  = open(app.root_path+"/static/dog_names.pkl", "rb")
    dog_names = pickle.load(dog_list)
    dog_list.close()
 
    #Load Model
    Resnet50_model = load_model(app.root_path+'/Resnet_50_20211004.h5')
    # summarize model.
    Resnet50_model.summary()

    # Extract Bottleneck Features [Code ref Udacity]
    def extract_Resnet50(tensor):
        return ResNet50(weights='imagenet', include_top=False, pooling='avg').predict(preprocess_input(tensor))


    def breed_predictor(img_path):
        '''Function that takes the file path of an image and returns the dog breed as predicted by a model.
        
        Input: Filepath to image 
        
        Output: Predicted Dog Breed     
        '''
        #Bottleneck from pre written function [ref Udacity]
        bottleneck_features = extract_Resnet50(path_to_tensor(img_path))
        # Add dimensions, problem being caused as Resnet50 in Keras has been updated, requiring average pooling layer. 
        bottleneck_features = np.expand_dims(bottleneck_features, axis=0)
        bottleneck_features = np.expand_dims(bottleneck_features, axis=0)

        #Make Prediction
        prediction = np.argmax(Resnet50_model.predict(bottleneck_features))
        return dog_names[(prediction)].split(".",1)[1]

    #https://stackoverflow.com/questions/51231576/tensorflow-keras-expected-global-average-pooling2d-1-input-to-have-shape-1-1

    # breed_predictor(r'D:\Christopher\Documents\Personal Project Files\dog_detector\app\resources\static\images\Labrador_retriever_06449.jpg')

    #Face Algo
    def face_detector(img_path):
        '''
        INPUT:
        OUTPUT:
        '''
        # extract pre-trained face detector
        face_cascade = cv2.CascadeClassifier(app.root_path+'/static/haarcascade_frontalface_alt.xml')
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    ##Dog Algo##
    def ResNet50_predict_labels(img_path):
    # define ResNet50 model
        ResNet50_model = ResNet50(weights='imagenet')
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img))

    ### returns "True" if a dog is detected in the image stored at img_path
    def dog_detector(img_path):
        prediction = ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151)) 

    # Dog detector algo. 
    def dog_predictor (img_path):
        ''' Function to take an image path and run it through the algorithm developed
        INPUT: File path to image
        OUTPUT: Classify Using Algorithm
        '''
        if dog_detector(img_path) == True:
            response = "This looks like a dog! I think it might be a"
            breed = breed_predictor(img_path)
            
        if face_detector(img_path) == True: 
            response = "This looks like a human face! If it were a dog though, I think it is a "
            breed =  breed_predictor(img_path)
        
        return f'{response} {breed}' 

    return dog_predictor(img_path)
    # dog_predictor(r'D:\Christopher\Documents\Personal Project Files\dog_detector\app\resources\static\images\Labrador_retriever_06449.jpg')