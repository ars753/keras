import numpy as np
from keras.applications import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import matplotlib.pyplot as plt


base_model = VGG19(weights='imagenet')

# выбираю слои для стиля
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer = 'block5_conv2'

# создаю модель для стиля контента
style_outputs = [base_model.get_layer(name).output for name in style_layers]
content_output = base_model.get_layer(content_layer).output
feature_extractor = Model(inputs=base_model.input, outputs=style_outputs + [content_output])

# эта функция обрабатывает изоброжение перед загрузкой
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# функция для отображения изображения
def show_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# функция для извлечения стилевых призанков
def extract_features(img_path):
    img = load_and_preprocess_image(img_path)
    outputs = feature_extractor.predict(img)
    style_features = outputs[:-1]
    content_feature = outputs[-1]
    return style_features, content_feature


style_image_path = 'golden-retriever.jpg'
content_image_path = 'drake.jpg'

style_features, _ = extract_features(style_image_path)
_, content_feature = extract_features(content_image_path)


content_img = load_and_preprocess_image(content_image_path)


show_image(content_img[0])




