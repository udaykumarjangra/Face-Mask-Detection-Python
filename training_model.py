from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from imutils import paths
import tensorflow as tf
import numpy as np
import os
data=np.load('data.npy')
labels=np.load('labels.npy')
(X_train, X_test, y_train, y_test)=train_test_split(data,labels, test_size=0.20, random_state=0)
augmen=ImageDataGenerator( zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, rotation_range=20, fill_mode="nearest")

mobile_net=MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
model=mobile_net.output
model=tf.keras.layers.AveragePooling2D(pool_size=(7,7))(model)
model=tf.keras.layers.Flatten()(model)
model=tf.keras.layers.Dense(128, activation="relu")(model)
model=tf.keras.layers.Dropout(0.5)(model)
model=tf.keras.layers.Dense(2, activation="softmax")(model)

history=Model(inputs=mobile_net.input, outputs=model)
for layer in mobile_net.layers:
    layer.trainable = False
history.compile(loss="binary_crossentropy", optimizer="Adam", metrics=['accuracy'])
fit=history.fit(augmen.flow(X_train, y_train, batch_size=32),
                steps_per_epoch=len(X_train)//32,
                validation_data=(X_test, y_test),
                validation_steps=len(X_test)//32,
                epochs=20)

#prediction=history.predict(X_test, batch_size=32)
#prediction=np.argmax(prediction, axis=1)
history.save('model', save_format='h5')

                


