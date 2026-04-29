from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR,"app","templates"),
    static_folder=os.path.join(BASE_DIR,"app","static")
)

UPLOAD_FOLDER = os.path.join(BASE_DIR,"app","static","uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# LOAD MODELS
models = {
    "cnn": tf.keras.models.load_model("models/cnn_model.keras"),
    "resnet": tf.keras.models.load_model("models/resnet_model.keras"),
    "fusion": tf.keras.models.load_model("models/fusion_model.keras")
}

def init_db():
    conn = sqlite3.connect("database.db")
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
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (filename, model, result, confidence, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """,(filename, model, result, conf, datetime.now()))
    conn.commit()
    conn.close()

@app.route("/", methods=["GET","POST"])
def index():
    result=""
    img_path=""

    if request.method=="POST":
        file = request.files["file"]
        model_name = request.form.get("model","fusion")

        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        img = load_img(path, target_size=(224,224))
        img = img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = models[model_name].predict(img)
        prob = float(pred[0][0])

        result = "Landslide" if prob < 0.5 else "Non-Landslide"

        save(result, model_name, file.filename, prob)

        img_path = "uploads/" + file.filename

    return render_template("index.html", result=result, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)