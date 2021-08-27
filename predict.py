import time
import json
import glob
import sys
import os.path
import argparse

import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)   
       
    
def process_image(image):
    
    ''' The process_image function should take in an image (in the form of a NumPy array) and return an image 
        in the form of a NumPy array with shape (224, 224, 3).'''
    
    image = np.squeeze(image)
    final_image = tf.cast(image, tf.float32)
    final_image = tf.image.resize(final_image, [224, 224])
    final_image /=255.0
    
    return final_image

def predict(image_path, model,class_names, topk = 5):
    
    ''' the predict function should take an image, a model, and then returns 
        the top 𝐾 most likely class labels along with the probabilities. '''
    
    image = Image.open(image_path)
    image = np.asarray(image)
    
    # process the image and add extra dimension
    image = process_image(image)
    preds = model.predict(np.expand_dims(image, 0))
    
    # get the top K flower codes, names and the probabilities
    class_codes = (tf.math.top_k(preds[0], k=topk, sorted=True, name=None)).indices.numpy().tolist()
    probs       = (tf.math.top_k(preds[0], k=topk, sorted=True, name=None)).values.numpy().tolist()
    
    flower_names = [class_names[str(code+1)] for  code in class_codes]
    probs   = [round(prob * 100, 2) for prob in probs]
    
    return flower_names, probs 

def main():
    """
        Flower Prediction Application
    """
   # define parser
    parser = argparse.ArgumentParser(description='Flower Prediction Application')
    
    # add arguments
    parser.add_argument('--model',
                         action = "store",
                         dest = 'model',
                         default = 'keras_model.h5',
                         help = 'path to model file'
                        ) 
    parser.add_argument('--category_names',
                         action = "store",
                         dest = 'class_names',
                         default = 'label_map.json',
                         help = 'path to json file with class names'
                        ) 
    parser.add_argument('--top_k',
                         action = "store",
                         dest = 'topk',
                         type = int,
                         default = 5,
                         help = 'Number of top records to print'
                       )
    # Getting additional parameters
    commands = parser.parse_args(sys.argv[2:])
    model    = commands.model
    classes  = commands.class_names
    top_k    = commands.topk
    
    # Load categories
    try:
        with open(classes, 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        print(e)
        sys.exit()
        
    # Load image and model
    try:
        image = str(sys.argv[1])
        
    except Exception as e1:
        print(e1)
        sys.exit()
        
    if os.path.isfile(image) == False:
        print('Image does not exist')
        sys.exit()
    if os.path.isfile(model) == False:
        print('Model does not exist')
        sys.exit()
    loaded_model = tf.keras.models.load_model(model, custom_objects = {'KerasLayer': hub.KerasLayer})                                   
    
    # Predict the flower name 
    flower_name = image.split('/')[len(image.split('/'))-1][:-4]
    results = predict(image, loaded_model, class_names, top_k)
    
    # Printing the result
    print("The original flower name is {}".format(flower_name))
    print("The top {} name prediction for the given flower image: {}".format(top_k, results[0])) 
    print("The top {} probabilities are: {}".format(top_k, results[1]))
 
if __name__ == '__main__':
    main()  
                                              
                                            