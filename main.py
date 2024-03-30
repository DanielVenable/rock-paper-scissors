import time
import os
import cv2

# Don't show tensorflow warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras


def get_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print('Unable to open camera')
        exit(1)

    return camera

camera = get_camera()
model = keras.models.load_model('model.keras')

def predict(image):
    # Make the image match what the model expects
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    return model.predict(image, verbose=0)

PAPER = 0
ROCK = 1
SCISSORS = 2

while True:
    success, image = camera.read()
    if success:
        prediction = predict(image)[0]
        if prediction[PAPER] > 0.5:
            print('Paper')
        if prediction[ROCK] > 0.8:
            print('Rock')
        if prediction[SCISSORS] > 0.5:
            print('Scissors')
    time.sleep(1)
