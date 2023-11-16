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


batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 54

train_len = 60700
train_dir = '/content/drive/MyDrive/Dataset(org)/'

def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0
    count=0
    for folderName in os.listdir(folder):
        count+=1
        print(count)
        if not folderName.startswith('.'):
            if folderName in ['sign_1']:
                label = 0
            elif folderName in ['sign_2']:
                label = 1
            elif folderName in ['sign_3']:
                label = 2
            elif folderName in ['sign_4']:
                label = 3
            elif folderName in ['sign_5']:
                label = 4
            elif folderName in ['sign_6']:
                label = 5
            elif folderName in ['sign_7']:
                label = 6
            elif folderName in ['sign_8']:
                label = 7
            elif folderName in ['sign_9']:
                label = 8
            elif folderName in ['sign_10']:
                label = 9
            elif folderName in ['sign_11']:
                label = 10
            elif folderName in ['sign_12']:
                label = 11
            elif folderName in ['sign_13']:
                label = 12
            elif folderName in ['sign_14']:
                label = 13
            elif folderName in ['sign_15']:
                label = 14
            elif folderName in ['sign_16']:
                label = 15
            elif folderName in ['sign_17']:
                label = 16
            elif folderName in ['sign_18']:
                label = 17
            elif folderName in ['sign_19']:
                label = 18
            elif folderName in ['sign_20']:
                label = 19
            elif folderName in ['sign_21']:
                label = 20
            elif folderName in ['sign_22']:
                label = 21
            elif folderName in ['sign_23']:
                label = 22
            elif folderName in ['sign_24']:
                label = 23
            elif folderName in ['sign_25']:
                label = 24
            elif folderName in ['sign_26']:
                label = 25
            elif folderName in ['sign_27']:
                label = 26
            elif folderName in ['sign_28']:
                label = 27
            elif folderName in ['sign_29']:
                label = 28
            elif folderName in ['sign_31']:
                label = 29
            elif folderName in ['sign_32']:
                label = 30
            elif folderName in ['sign_33']:
                label = 31
            elif folderName in ['sign_34']:
                label = 32
            elif folderName in ['sign_35']:
                label = 33
            elif folderName in ['sign_36']:
                label = 34
            elif folderName in ['sign_37']:
                label = 35
            elif folderName in ['sign_38']:
                label = 36
            elif folderName in ['sign_39']:
                label = 37
            elif folderName in ['sign_40']:
                label = 38
            elif folderName in ['sign_41']:
                label = 39
            elif folderName in ['sign_42']:
                label = 40
            elif folderName in ['sign_43']:
                label = 41
            elif folderName in ['sign_44']:
                label = 42
            elif folderName in ['sign_45']:
                label = 43
            elif folderName in ['sign_46']:
                label = 44
            elif folderName in ['sign_47']:
                label = 45
            elif folderName in ['sign_48']:
                label = 46
            elif folderName in ['sign_49']:
                label = 47
            elif folderName in ['sign_50']:
                label = 48
            elif folderName in ['sign_51']:
                label = 49
            elif folderName in ['sign_52']:
                label = 50
            elif folderName in ['sign_53']:
                label = 51
            elif folderName in ['sign_54']:
                label = 52
            elif folderName in ['sign_55']:
                label = 53

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
y_cat_train = to_categorical(y_train,54)
y_cat_test = to_categorical(y_test,54)

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

model.add(Dense(54, activation='softmax'))

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