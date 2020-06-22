import cv2
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import config
import matplotlib.pyplot as plt
import os

def get_filepath():
    '''Grab the image the user wrote'''
    Ap = argparse.ArgumentParser()
    Ap.add_argument("-f", "--file", required=True, help="Path to the image")
    Ap.add_argument("-t", "--text", help="Path to the text file (Not required)")
    ima = vars(Ap.parse_args())['file']
    txt = vars(Ap.parse_args())['text']
    return ima, txt

def ready_image(filepath, thresh_val):
    '''Loads the image and returns the image, threshold, edged image and a list of detected contours'''
    #read and load the image
    img = cv2.imread(filepath, 0)
    img = cv2.bitwise_not(img)

    #thresh the image
    thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV) #TODO make thresh into avg

    #Get the edges
    edged = cv2.Canny(img, 200, 200)

    #Find all contour borders
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    return img, thresh, edged, contours

def load_the_model(model_path):
    '''loads and returns the model'''
    model = load_model(model_path)
    # os.system('cls') #Clear the screen again
    return model

def recognise_contours(img, contours, model):
    '''Recognises the images in a given list of contours'''

    #Define the final string the model will predict
    string = ''

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt) #Get the bounding box of the detected contours
        roi = img[y-3 : y+h + 3, x-3 : x+w + 3] #define the contoured image as a region of image and give it a little space of 3 px on each side

        roi = cv2.resize(roi, (28, 28))
        
        roi = np.reshape(roi, [-1, 28, 28, 1])

        string += config.alphabets[np.argmax(model.predict(roi))]

    return string

def recog(img, model):
    img = cv2.resize(img, (28, 28))

    img = np.reshape(img, [-1, 28, 28, 1])

    return config.alphabets[np.argmax(model.predict(img))]

def get_correct():
    pass

path, text = get_filepath()

if text is None:
    correct = input("Type the correct string: ") #Ask the user for the correct string
else:
    correct = open(text, 'r').read().capitalize()

#NOTE You can make the program mess with the images and predict for each. Append these letter to a list. Then, chose the element which is the most common

img, thresh, edged, contours = ready_image(path, 200)

model = load_the_model(config.model_path)

final = recog(img, model)
os.system('cls')

score = 0

for prdl, corl in final, correct:
    if prdl == corl:
        score += 1

accuracy = round((score / len(correct)) * 100, 5)

print(\
f"""
Predicted : {final}
Correct   : {correct}
Accuracy  : {accuracy} %""")