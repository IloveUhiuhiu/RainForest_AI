import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model


base_model = MobileNetV2(weights='imagenet', include_top=False,
input_shape=(224, 224, 3))

model = load_model('weights/best_model.h5')

def predict_image(x):
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = base_model.predict(x)
    predictions = model.predict(y)
    print(f"predictions: {predictions}")
    return np.argmax(predictions[0])  
   
