from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os

from utils.video_processing import extract_landmarks
from utils.audio_processing import extract_mfcc
from utils.visualization import generate_probability_chart

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model("deepfake_model.h5")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    file = request.files["video"]

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(path)

    landmarks = extract_landmarks(path)

    mfcc = extract_mfcc(path)

    if landmarks is None or mfcc is None:
        return "Processing failed"

    features = np.concatenate((landmarks, mfcc))

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0][0]

    fake_prob = float(prediction)
    real_prob = 1 - fake_prob

    chart_path = "static/outputs/chart.png"

    generate_probability_chart(real_prob, fake_prob, chart_path)

    if fake_prob > 0.5:
        result = "FAKE"
    else:
        result = "REAL"

    return render_template(
        "result.html",
        result=result,
        real=round(real_prob*100,2),
        fake=round(fake_prob*100,2),
        chart=chart_path
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
