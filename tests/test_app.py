from io import BytesIO

from PIL import Image

from app.app import app

client = app.test_client()

PATIENT = {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
           "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
           "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7}


def test_pages_load():
    for url in ["/", "/heart", "/skin"]:
        assert client.get(url).status_code == 200


def test_heart_form_returns_risk():
    page = client.post("/heart", data=PATIENT).get_data(as_text=True)
    assert "Estimated probability" in page


def test_heart_form_rejects_bad_input():
    page = client.post("/heart", data={"age": "abc"}).get_data(as_text=True)
    assert "valid number" in page


def test_skin_upload_returns_prediction():
    buf = BytesIO()
    Image.new("RGB", (600, 450), (180, 120, 100)).save(buf, "JPEG")
    buf.seek(0)
    page = client.post("/skin", data={"image": (buf, "lesion.jpg")},
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "Predicted lesion type" in page


def test_skin_rejects_non_image():
    page = client.post("/skin", data={"image": (BytesIO(b"not an image"), "x.jpg")},
                       content_type="multipart/form-data").get_data(as_text=True)
    assert "valid image" in page