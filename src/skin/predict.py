import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/skin_model.pt"
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

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
_model = None


def load_model():
    global _model
    if _model is None:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _model = m
    return _model


def predict_skin(image: Image.Image) -> dict:
    """Take a PIL image, return the predicted lesion type."""
    x = _tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(load_model()(x), dim=1)[0]
    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda p: p[1], reverse=True)
    top = ranked[0][0]
    return {
        "prediction": top,
        "name": NAMES[top],
        "confidence": round(ranked[0][1], 3),
        "probabilities": {c: round(p, 3) for c, p in ranked},
    }