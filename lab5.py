import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from PIL import Image


model = keras.applications.VGG16(weights='imagenet')


img_path = 'drake.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


preds = model.predict(x)


decoded_preds = decode_predictions(preds, top=1)[0]
print('Predicted class:', decoded_preds[0][1])
