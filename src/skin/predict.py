import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = "models/skin_model.onnx"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NAMES = {
    "akiec": "Actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (mole)",
    "vasc": "Vascular lesion",
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None


def load_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def run_model(image: Image.Image):
    """Return (logits, class activation maps) for one image."""
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    logits, cams = load_session().run(None, {"image": x})
    return logits[0], cams[0]


def predict_skin(image: Image.Image) -> dict:
    """Take a PIL image, return the predicted lesion type."""
    logits, _ = run_model(image)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda p: p[1], reverse=True)
    top = ranked[0][0]
    return {
        "prediction": top,
        "name": NAMES[top],
        "confidence": round(ranked[0][1], 3),
        "probabilities": {c: round(p, 3) for c, p in ranked},
    }