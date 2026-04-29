import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Flatten
from tensorflow.keras.models import Model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_PATH, exist_ok=True)

IMG_SIZE = (224,224)
BATCH_SIZE = 4
EPOCHS = 10

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

input_layer = Input(shape=(224,224,3))

# CNN branch
cnn = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(input_layer)
cnn = tf.keras.layers.MaxPooling2D(2,2)(cnn)
cnn = tf.keras.layers.Conv2D(64,(3,3),activation='relu')(cnn)
cnn = tf.keras.layers.MaxPooling2D(2,2)(cnn)
cnn = Flatten()(cnn)

# ResNet branch
resnet = ResNet101(weights='imagenet', include_top=False, input_tensor=input_layer)
resnet.trainable = False
resnet_out = GlobalAveragePooling2D()(resnet.output)

# Fusion
fusion = Concatenate()([cnn, resnet_out])

x = Dense(128, activation='relu')(fusion)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=EPOCHS)

model.save(os.path.join(MODEL_PATH, "fusion_model.keras"))

print("✅ Fusion model saved")