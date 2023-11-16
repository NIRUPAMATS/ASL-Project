# import tensorflowjs as tfjs
import tensorflow as tf
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("ASL.h5", compile=False) 
# tfjs.converters.save_keras_model(model, "tfjs_model")
# Save the model
tf.saved_model.save(model, 'my_model')

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Define the true and predicted labels arrays
true_labels = []
predicted_labels = []

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    # confidence_score = prediction[0][index]

    # Add the true and predicted labels to their respective arrays
    true_labels.append(class_name)
    predicted_labels.append(class_names[index])

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(50)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

# Compute the confusion matrix
# confusion = confusion_matrix(true_labels, predicted_labels, labels=class_names)
camera.release()
cv2.destroyAllWindows()