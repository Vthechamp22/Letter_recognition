import cv2
import numpy as np
import argparse
import config
import os
import tensorflow as tf; os.system('cls') # clear the screen because when tensorflow imports, it prints cudlart error
import pickle
import time
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
# from tensorflow.keras import Sequential

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
    '''loads and returns the model'''
    model = tf.keras.models.load_model(model_path)
    os.system('cls') #Clear the screen again
    return model

def recognise_contours(img, contours, model, model_con):
    '''Recognises the images in a given list of contours'''

    #Define the final string the model will predict
    string = ''

    #Make a list for all the images detected so that the model can learn later on
    letter_rois = []

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt) #Get the bounding box of the detected contours
        roi = img[y-3 : y+h + 3, x-3 : x+w + 3] #define the contoured image as a region of image and give it a little space of 3 px on each side
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #convert it to grayscale

        #Configure the image according to the model. 
        if model_con == config.Dataset2:
            roi = cv2.bitwise_not(roi)
            roi = cv2.resize(roi, (100, 100))
            letter_rois.append(roi)
            roi = roi.reshape([1, 100, 100, 1])
        else:
            roi = cv2.resize(roi, (28, 28))
            letter_rois.append(roi)
            roi = roi.reshape([1, 28, 28, 1])
        string += config.alphabets[np.argmax(model.predict(roi))]

    return letter_rois, string

def load_previous_data(model_con):
    #Load the previous data and labels
    if model_con == config.Dataset1:
        with open('./Dataset1_X.pickle', 'rb') as f:
            X = pickle.load(f)

        with open('./Dataset1_y.pickle', 'rb') as f:
            y = pickle.load(f)

    else:
        with open('./Dataset2_X.pickle', 'rb') as f:
            X = pickle.load(f)

        with open('./Dataset2_y.pickle', 'rb') as f:
            y = pickle.load(f)
    
    return X, y

def get_correct_string():
    '''Ask the user for the correct string for the accuracy.'''
    correct = input("Please type the correct string: ").upper() #Ask the user for the correct string for testing accuracy.
    sure = input("Are you sure? (y/n)").lower() # Confirm the sentence
    if sure == "n":
        get_correct_string()
    return correct

def calc_acc(correct, string, X, y, letter_rois, model_con, img):
    score = 0

    for i in range(len(correct)):
        try:
            if string[i] == correct[i]:
                score += 1
            roi = np.array(letter_rois[i])
            if model_con == config.Dataset2:
                roi = cv2.resize(roi, (100, 100))
                np.append(X, roi.reshape([1, 100, 100, 1]))
            else:
                roi = cv2.resize(roi, (28, 28))
                np.append(X, roi.reshape([1, 28, 28, 1]))
            np.append(y, config.alphabets.index(correct[i]))
        except IndexError:
            pass

    acc = (score/len(string)) * 100

    if model_con == config.Dataset2:
        X = X.reshape([-1, 100, 100, 1])
    else:
        X = X.reshape([-1, 28, 28, 1])
    return acc, score, X, y

def replace(model, X, y):
    # Replace the old pickle files with the new data
    if model_con == config.Dataset2:
        with open('./Dataset1_X.pickle', 'wb') as f:
            pickle.dump(X, f)
        with open('./Dataset1_y.pickle', 'wb') as f:
            pickle.dump(y, f)

    else:
        with open('./Dataset2_X.pickle', 'wb') as f:
            pickle.dump(X, f)

        with open('./Dataset2_y.pickle', 'wb') as f:
            pickle.dump(y, f)

def compile_and_retrain(model, X, y, model_con):
    X = np.array(X)
    y = np.array(y)

    if model_con == config.Dataset2:
        img_size = 100
    else:
        img_size = 28

    y = y.reshape([-1, 1])
    X = X.reshape([-1, img_size, img_size, 1])

    # Finally compile the model and train it and save it
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'],)
    model.fit(X, y, epochs=3)# verbose=0) #TODO decide whether showing the user the progress is better or not (verbose)
    model.save("retrained.model")

# get the filepath of the image
filepath = get_filepath()

# get the image, thresh, edged and the contours for the image
img, thresh, edged, contours = ready_image(filepath, 200)

# Inform the user how many letters were detected
print(f"Detected {str(len(contours))} letters")

# Load the model
model_con = config.retrained
model = load_model(model_con)

# Get the region of images for the letters and the recognised string
letter_rois, string = recognise_contours(img, contours, model=model, model_con=model_con)

# Loads the previous x and y data
X, y = load_previous_data(model_con)

#let the user know the no. of letters detected
correct = get_correct_string()

# get the accuracy and score
acc, score, X, y = calc_acc(correct, string, X, y, letter_rois, model_con, img)

# replaces the X and y data
replace(model_con, X, y)
    
#Reveal the accuracy, what the model predicted, and the real string.
if acc < 50.0:
    print(f"Oops! I predicted {string} (Which is way off)\nAccuracy: {round(acc, 4)} %.\nCorrect are {score} out of {len(string)}")
else:
    print(f"Okay... I predicted {string} (Which not thaaaat way off hopefully)\nAccuracy: {round(acc, 4)} %.\n Correct are {score} out of {len(string)}")

time.sleep(1)

#Inform the user to not kill the program as it will train.
print("As this is a self learning model, the model will now train again. Please leave the program running and don't kill it")

# Train the model
compile_and_retrain(model, X, y, model_con)

# inform the user that the model has finished training and they can now close and exit
print('Okay done! Good to go! Thanks for keeping patient! Hopefully I will predict better next time...')