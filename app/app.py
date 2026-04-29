from flask import Flask, render_template, request
import numpy as np
from PIL import Image
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

print("BASE_DIR:", BASE_DIR)

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
        file = request.files["file"]

        if file.filename == "":
            return "❌ No file selected"

        # Save image
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        # 🖼 Simple preprocessing (no ML)
        img = Image.open(path).resize((224, 224))
        img_array = np.array(img) / 255.0

        # 🔮 Dummy prediction (temporary)
        prob = float(np.mean(img_array))  # simple logic

        result = "Landslide" if prob < 0.5 else "Non-Landslide"

        # 💾 Save to DB
        save(result, "demo", file.filename, prob)

        img_path = "uploads/" + file.filename

    return render_template("index.html", result=result, img_path=img_path)

# ▶️ Run App
if __name__ == "__main__":
    app.run(debug=True)