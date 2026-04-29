import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("models/landslide_fusion_model.keras")

IMG_SIZE = (224,224)

# Data generator
datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Predict
predictions = model.predict(data)

# Convert probabilities to 0/1 labels
y_pred = (predictions > 0.5).astype(int).reshape(-1)

y_true = data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))