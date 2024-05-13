
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from skimage.transform import resize
import seaborn as sns
import cv2

cifar100 = tf.keras.datasets.cifar100

(x_train, y_train) , (x_test, y_test) = cifar100.load_data(label_mode="fine")

print("train shape: ", x_train.shape)
print("test shape: ", x_test.shape)

classes_to_keep = [12, 15, 54, 61, 66, 68, 77]

train_mask = np.isin(y_train, classes_to_keep).reshape(-1)
test_mask = np.isin(y_test, classes_to_keep).reshape(-1)

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

class_names = {
    12: 'bridge',
    15: 'camel',
    54: 'orchid',
    61: 'plate',
    66: 'racoon',
    68: 'road',
    77: 'snail'
}

class_names[int(y_train[0][0])]

plt.figure(figsize=(5,5))
for i in range(12):
    # Create a subplot for each image
    plt.subplot(3, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Display the image
    plt.imshow(x_train[i])

    # Set the label as the title
    plt.title(class_names[y_train[i][0]], fontsize=12)

# Display the figure
plt.show()

print("train shape: ", x_train.shape)
print("test shape: ", x_test.shape)

y_train = tf.one_hot(y_train,
                     depth=y_train.max() + 1,
                     dtype=tf.float64)
y_test = tf.one_hot(y_test,
                   depth=y_test.max() + 1,
                   dtype=tf.float64)

y_train = tf.squeeze(y_train)
y_test = tf.squeeze(y_test)

from keras import layers

model = tf.keras.models.Sequential()

# Consider a slightly larger input size if feasible
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.25))  # Reduced dropout slightly

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.45))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
# Removed a MaxPooling layer here
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(78, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.summary()

#%%
hist = model.fit(x_train, y_train,
                 epochs=15,
                 batch_size=64,
                 verbose=1,
                 validation_data=(x_test, y_test))

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()

loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(loss))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc='upper left')
plt.show()

pred_cnn = model.predict(x_test)
predicted = np.argmax(pred_cnn, axis=1)

plt.figure(figsize=(15, 15))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    true_label = class_names[np.argmax(y_test[i])]  # Get true label using argmax
    predicted = class_names[np.argmax(model.predict(x_test)[i])]  # Get predicted label
    if true_label == predicted:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel(f"True: {true_label}\nPred (MLP): {predicted}", color=color)
plt.tight_layout()
plt.show()
