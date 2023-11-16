# import data processing and visualisation libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# import image processing libraries
import cv2
import skimage
from skimage.transform import resize

# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
import os

print("Packages imported...")

batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 9

train_len = 18000
train_dir = '/content/drive/MyDrive/Dataset/'

def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['fever']:
                label = 0
            elif folderName in ['stomachache']:
                label = 1
            elif folderName in ['headache']:
                label = 2
            elif folderName in ['cold']:
                label = 3
            elif folderName in ['cough']:
                label = 4
            elif folderName in ['sneeze']:
                label = 5
            elif folderName in ['sprain']:
                label = 6
            elif folderName in ['yes']:
                label = 7
            elif folderName in ['no']:
                label = 8
            
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))

                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
    return X,y
X_train, y_train = get_data(train_dir)
print("Images successfully imported...")

print("The shape of X_train is : ", X_train.shape)
print("The shape of y_train is : ", y_train.shape)

print("The shape of one image is : ", X_train[0].shape)

plt.imshow(X_train[0])
plt.show()





X_data = X_train
y_data = y_train
print("Copies made...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)

# One-Hot-Encoding the categorical data
from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train,9)
y_cat_test = to_categorical(y_test,9)

# Checking the dimensions of all the variables
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_cat_train.shape)
print(y_cat_test.shape)

# This is done to save CPU and RAM space while working on Kaggle Kernels. This will delete the specified data and save some space!
import gc
del X_data
del y_data
gc.collect()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
print("Packages imported...")

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(9, activation='softmax'))

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_cat_train,
          epochs=15,
          batch_size=64,
          verbose=2,
          validation_data=(X_test, y_cat_test),
         callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)
print("The model metrics are")
# metrics

metrics[['loss','val_loss']].plot()
plt.show()

metrics[['accuracy','val_accuracy']].plot()
plt.show()

model.evaluate(X_test,y_cat_test,verbose=0)

predictions = model.predict(X_test)
classes_x=np.argmax(predictions,axis=1)
print("Predictions done...")

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,classes_x))

plt.figure(figsize=(12,12))
sns.heatmap(confusion_matrix(y_test,classes_x))
plt.show()

from keras.models import load_model
model.save('ASL.h5')
print("Model saved successfully...")


