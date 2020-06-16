from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils import paths
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

dataset=list(paths.list_images('dataset'))
data=[]
labels=[]

for i in dataset:
    label=i.split(os.path.sep)[-2]
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)
    labels.append(label)

data=np.array(data,dtype="float")
labels=np.array(labels)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)
np.save('data',data)
np.save('labels',labels)


                

