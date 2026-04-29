import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
import os

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------------
# PARAMETERS
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# -----------------------------
# INPUT LAYER
# -----------------------------
input_layer = Input(shape=(224,224,3))

# -----------------------------
# CNN BRANCH
# -----------------------------
x = Conv2D(32,(3,3),activation='relu')(input_layer)
x = MaxPooling2D(2,2)(x)

x = Conv2D(64,(3,3),activation='relu')(x)
x = MaxPooling2D(2,2)(x)

x = Flatten()(x)
cnn_features = Dense(128,activation='relu')(x)

# -----------------------------
# RESNET101 BRANCH
# -----------------------------
resnet_base = ResNet101(
    weights='imagenet',
    include_top=False,
    input_tensor=input_layer
)

resnet_base.trainable = False

y = resnet_base.output
y = GlobalAveragePooling2D()(y)
resnet_features = Dense(128,activation='relu')(y)

# -----------------------------
# FEATURE FUSION
# -----------------------------
fusion = Concatenate()([cnn_features, resnet_features])

z = Dense(64,activation='relu')(fusion)
output = Dense(1,activation='sigmoid')(z)

model = Model(inputs=input_layer, outputs=output)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(os.path.join(MODEL_PATH,"landslide_fusion_model.keras"))

print("Fusion Model training completed successfully")
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()