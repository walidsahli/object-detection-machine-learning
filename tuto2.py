import tensorflow as tf
import skimage
import cv2
import os
import matplotlib.pyplot as plt
from skimage import transform 
from tensorflow import keras
from skimage.color import rgb2gray
import numpy as np
import time 
import shutil

names = []

camera = cv2.VideoCapture(0)
number_of_persons = input("How mane faces to scan ?") 
i = 0
time.sleep(5)
j = 0

os.mkdir('images')
os.mkdir('images/test')
os.mkdir('images/train')

while i <= int(number_of_persons)*200 - 1 :
    if i % 200 == 0:
        if i > 0 :
            j = j + 1
        os.mkdir('images/test/{}'.format(j))
        os.mkdir('images/train/{}'.format(j))
        name = input('give a name please : ')
        print('scaning {}"s face ...'.format(name))
        names.append(name)
    ret, frame = camera.read()
    file_name = 'image_{}.jpg'.format(i)
    cv2.imwrite(file_name,frame)
    cv2.waitKey(1)
    print(i)
    if i % 200 < 160:
        shutil.move(file_name,'images/train/{}/{}'.format(j,file_name))
    else:
        shutil.move(file_name,'images/test/{}/{}'.format(j,file_name))
    i = i + 1
camera.release()
cv2.destroyAllWindows()



def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(cv2.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "./images/"
train_data_directory = os.path.join(ROOT_PATH, "train")
test_data_directory = os.path.join(ROOT_PATH, "test")

#################################
# fit train images              #
#################################
train_images , train_labels = load_data(train_data_directory)
train_images = [transform.resize(image, (28, 28)) for image in train_images]
train_images = np.array(train_images)
train_images = rgb2gray(train_images)

#################################
# fit test images               #
#################################
test_images, test_labels = load_data(test_data_directory)
test_images = [transform.resize(image, (28, 28)) for image in test_images]
test_images = np.array(test_images)
test_images = rgb2gray(test_images)

#################################
# build model                   #
#################################
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # input shape isthe shape of images after they has been converted to 28x28 array
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#################################
# train model                   #
#################################
model.fit(train_images, train_labels, epochs=15)

#################################
# test model                    #
#################################

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc * 100  ,' %')

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


##### automate gaming #### 
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

chrome_options = webdriver.ChromeOptions()
prefs = {"profile.default_content_setting_values.notifications" : 2}
chrome_options.add_experimental_option("prefs",prefs)

browser = webdriver.Chrome(executable_path = r'./chromedriver' ,chrome_options=chrome_options)
browser.get("https://facebook.com")



# start predection from webcam images
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    arr = []
    arr.append(frame)
    arr = [transform.resize(image, (28, 28)) for image in arr]

    arr = np.array(arr)

    # Convert `images28` to grayscale
    arr = rgb2gray(arr)
    predictions = probability_model.predict(arr)
    res = np.argmax(predictions[0])
    cv2.putText(frame, names[res], (200, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0))
    cv2.imshow('current object',frame)
    cv2.waitKey(1)
    if names[res] == "up":
        browser.find_element_by_tag_name('body').send_keys(Keys.ARROW_UP)
    elif names[res] == "nothing":
        pass
    print(names[res])

camera.release()
cv2.destroyAllWindows()