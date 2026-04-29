import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["cnn","resnet","fusion"])
parser.add_argument("--image", default="test.jpg")
args = parser.parse_args()

model_paths = {
    "cnn": "models/cnn_model.keras",
    "resnet": "models/resnet_model.keras",
    "fusion": "models/fusion_model.keras"
}

model = tf.keras.models.load_model(model_paths[args.model])

img = load_img(args.image, target_size=(224,224))
img = img_to_array(img)/255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
prob = pred[0][0]

print("Prediction:", prob)

# 🔥 FIXED LOGIC
if prob < 0.5:
    print("Result: Landslide")
else:
    print("Result: Non-Landslide")