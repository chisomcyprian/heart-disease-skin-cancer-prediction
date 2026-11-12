from PIL import Image
from src.skin.predict import predict_skin, CLASSES


def test_skin_prediction_format():
    img = Image.new("RGB", (600, 450), (180, 120, 100))
    result = predict_skin(img)
    assert result["prediction"] in CLASSES
    assert abs(sum(result["probabilities"].values()) - 1) < 0.01