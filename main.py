import time
import os
import cv2
import numpy as np
import warnings

# Don't show any warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras

tf.get_logger().setLevel('ERROR')

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

# A model that predicts what the user will pick next and acts accordingly
class Player():
    history = []
    memory_length = 3
    scores = [0, 0]

    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Input((3,3), 1),
            keras.layers.Conv1D(64, 10, activation='relu', padding='same'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='sigmoid')
        ])

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy')

    def next(self, user_action):
        if len(self.history) >= self.memory_length:
            buf = tf.convert_to_tensor([self.history[-self.memory_length:]]) # Last few elements of the history

            # Make a prediction
            prediction, = self.model(buf, training=False)

            computer_action = self.decide(prediction)

            # Learn from the new information
            self.model.train_on_batch(buf, result_tensors[user_action])
        else:
            computer_action = np.random.randint(3)

        self.history.append(one_hot[user_action])

        print(f'You picked {labels[user_action]}, I picked {labels[computer_action]}')

        winner = (computer_action - user_action) % 3
        if winner == 0:
            print('Tie!')
        if winner == 1:
            print('You won!')
            self.scores[0] += 1
        if winner == 2:
            print('I won!')
            self.scores[1] += 1

        print(f'Scores: You: {self.scores[0]}, Me: {self.scores[1]}\n')

    def decide(self, prediction):
        paper, rock, scissors = prediction
        probabilities = [rock - scissors, scissors - paper, paper - rock]
        return np.argmax(probabilities)

labels = ['Paper', 'Rock', 'Scissors']

PAPER = 0
ROCK = 1
SCISSORS = 2

one_hot = [
    [1, 0, 0], # Paper
    [0, 1, 0], # Rock
    [0, 0, 1]  # Scissors
]

result_tensors = list(map(lambda x: tf.convert_to_tensor([x]), one_hot))

def get_user_action():
    success, image = camera.read()
    if success:
        prediction, = predict(image)
        if prediction[PAPER] > 0.8:
            return PAPER
        if prediction[ROCK] > 0.8:
            return ROCK
        if prediction[SCISSORS] > 0.8:
            return SCISSORS
    return None

player = Player()

print('Ready!')

while True:
    action = get_user_action()
    if action is not None:
        player.next(action)
        time.sleep(0.5)