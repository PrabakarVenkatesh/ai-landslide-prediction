from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import sqlite3
from datetime import datetime

# 📁 BASE DIRECTORY
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🚀 Flask App Config
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "app", "templates"),
    static_folder=os.path.join(BASE_DIR, "app", "static")
)

# 📁 Upload Folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 📁 Model Directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 🔍 DEBUG INFO (VERY IMPORTANT)
print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)

if os.path.exists(MODEL_DIR):
    print("Files in models folder:", os.listdir(MODEL_DIR))
else:
    print("❌ Models folder NOT FOUND")

# 🤖 Load Models Safely
models = {}

def load_models():
    try:
        models["cnn"] = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
        print("✅ CNN model loaded")

    except Exception as e:
        print("❌ CNN model error:", e)

    try:
        models["resnet"] = tf.keras.models.load_model(os.path.join(MODEL_DIR, "resnet_model.keras"))
        print("✅ ResNet model loaded")

    except Exception as e:
        print("❌ ResNet model error:", e)

    try:
        models["fusion"] = tf.keras.models.load_model(os.path.join(MODEL_DIR, "fusion_model.keras"))
        print("✅ Fusion model loaded")

    except Exception as e:
        print("❌ Fusion model error:", e)

load_models()

# 📁 Database Path
DB_PATH = os.path.join(BASE_DIR, "database.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        model TEXT,
        result TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def save(result, model, filename, conf):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (filename, model, result, confidence, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (filename, model, result, conf, datetime.now()))
    conn.commit()
    conn.close()

# 🌐 Main Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    img_path = ""

    if request.method == "POST":
        file = request.files["file"]
        model_name = request.form.get("model", "fusion")

        if model_name not in models:
            return f"❌ Model '{model_name}' not loaded"

        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        # 🖼 Image preprocessing
        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # 🔮 Prediction
        model = models[model_name]
        pred = model.predict(img)
        prob = float(pred[0][0])

        result = "Landslide" if prob < 0.5 else "Non-Landslide"

        # 💾 Save to DB
        save(result, model_name, file.filename, prob)

        img_path = "uploads/" + file.filename

    return render_template("index.html", result=result, img_path=img_path)

# ▶️ Run App
if __name__ == "__main__":
    app.run(debug=True)