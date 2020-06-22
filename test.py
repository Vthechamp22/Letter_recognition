import cv2
import numpy as np
import argparse
import config
import os
import tensorflow as tf; os.system('cls') #clear the screen because when tensorflow imports, it prints cudlart error

def get_filepath():
    '''Grab the image the user wrote'''
    Ap = argparse.ArgumentParser()
    Ap.add_argument("-f", "--file", required=True, help="Path to the image")
    return vars(Ap.parse_args())['file']

def ready_image(filepath, thresh_val):
    '''Loads the image and returns the image, threshold, edged image and a list of detected contours'''
    #read and load the image
    img = cv2.imread(filepath)

    #thresh the image
    thresh = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV) #TODO make thresh into avg

    #Get the edges
    edged = cv2.Canny(img, 200, 200)

    #Find all contour borders
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    return img, thresh, edged, contours

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

img = get_filepath()

model = load_model(config.retrained)

img, thresh, edged, contours = ready_image(img, 200)

string = ''

for cnt in contours:
    [x,y,w,h] = cv2.boundingRect(cnt) #Get the bounding box of the detected contours
    roi = img[y-3 : y+h + 3, x-3 : x+w + 3] #define the contoured image as a region of image and give it a little space of 3 px on each side
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #convert it to grayscale

    roi = cv2.bitwise_not(roi)
    roi = cv2.resize(roi, (100, 100))
    roi = roi.reshape([1, 100, 100, 1])

    string += config.alphabets[np.argmax(model.predict(roi))]

print(string)