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

print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)

# 🤖 Load SINGLE Model (Best for deployment)
try:
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "fusion_model.keras")
    )
    print("✅ Fusion model loaded successfully")
except Exception as e:
    print("❌ Model loading error:", e)
    model = None

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

def save(result, model_name, filename, conf):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (filename, model, result, confidence, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (filename, model_name, result, conf, datetime.now()))
    conn.commit()
    conn.close()

# 🌐 Main Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    img_path = ""

    if request.method == "POST":
        if model is None:
            return "❌ Model not loaded. Check logs."

        file = request.files["file"]

        if file.filename == "":
            return "❌ No file selected"

        # Save image
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        # 🖼 Image preprocessing
        img = load_img(path, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # 🔮 Prediction
        pred = model.predict(img)
        prob = float(pred[0][0])

        result = "Landslide" if prob < 0.5 else "Non-Landslide"

        # 💾 Save to DB
        save(result, "fusion", file.filename, prob)

        img_path = "uploads/" + file.filename

    return render_template("index.html", result=result, img_path=img_path)

# ▶️ Run App (Local only)
if __name__ == "__main__":
    app.run(debug=True)