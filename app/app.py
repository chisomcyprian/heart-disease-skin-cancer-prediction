import base64
from io import BytesIO

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from src.heart.predict import predict_heart
from src.skin.gradcam import gradcam
from src.skin.predict import NAMES, predict_skin

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # reject uploads over 8 MB

YES_NO = [(0, "No"), (1, "Yes")]

# One entry per form field: name, label, then either choices or a number range
HEART_FIELDS = [
    {"name": "age", "label": "Age (years)", "min": 18, "max": 100, "step": 1},
    {"name": "sex", "label": "Sex", "choices": [(1, "Male"), (0, "Female")]},
    {"name": "cp", "label": "Chest pain type", "choices": [
        (1, "Typical angina"), (2, "Atypical angina"),
        (3, "Non-anginal pain"), (4, "No chest pain")]},
    {"name": "trestbps", "label": "Resting blood pressure (mm Hg)", "min": 80, "max": 220, "step": 1},
    {"name": "chol", "label": "Cholesterol (mg/dl)", "min": 100, "max": 600, "step": 1},
    {"name": "fbs", "label": "Fasting blood sugar above 120 mg/dl", "choices": YES_NO},
    {"name": "restecg", "label": "Resting ECG", "choices": [
        (0, "Normal"), (1, "ST-T wave abnormality"), (2, "Left ventricular hypertrophy")]},
    {"name": "thalach", "label": "Maximum heart rate reached", "min": 60, "max": 220, "step": 1},
    {"name": "exang", "label": "Chest pain during exercise", "choices": YES_NO},
    {"name": "oldpeak", "label": "ST depression during exercise", "min": 0, "max": 7, "step": 0.1},
    {"name": "slope", "label": "ST segment slope", "choices": [
        (1, "Upsloping"), (2, "Flat"), (3, "Downsloping")]},
    {"name": "ca", "label": "Major vessels seen on imaging", "choices": [
        (0, "0"), (1, "1"), (2, "2"), (3, "3")]},
    {"name": "thal", "label": "Thallium stress test", "choices": [
        (3, "Normal"), (6, "Fixed defect"), (7, "Reversible defect")]},
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    result, error, values = None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            patient = {f["name"]: float(values[f["name"]]) for f in HEART_FIELDS}
            result = predict_heart(patient)
                        # Never show 0% or 100%: no model is that certain
            result["percent"] = min(max(round(result["probability"] * 100), 1), 99)
        except (KeyError, ValueError):
            error = "Please fill in every field with a valid number."
    return render_template("heart.html", fields=HEART_FIELDS,
                           result=result, error=error, values=values)




def to_data_uri(image):
    """Turn an image into text the browser can display, with no file saved."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/skin", methods=["GET", "POST"])
def skin():
    result, error = None, None
    if request.method == "POST":
        file = request.files.get("image")
        try:
            image = Image.open(file.stream).convert("RGB")
            image.thumbnail((600, 600))
            result = predict_skin(image)
            result["percent"] = min(round(result["confidence"] * 100), 99)
            result["rows"] = [(NAMES[c], round(p * 100))
                              for c, p in result["probabilities"].items()]
            result["original"] = to_data_uri(image)
            result["heatmap"] = to_data_uri(gradcam(image))
        except (AttributeError, UnidentifiedImageError, OSError):
            error = "Please upload a valid image file (JPG or PNG)."
    return render_template("skin.html", result=result, error=error)



if __name__ == "__main__":
    app.run(debug=True)