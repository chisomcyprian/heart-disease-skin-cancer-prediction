import numpy as np
from PIL import Image

from src.skin.predict import run_model


def _jet(v):
    """Blue-to-red colour map for values between 0 and 1."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam(image: Image.Image) -> Image.Image:
    """Return the image overlaid with a heatmap of where the model looked."""
    image = image.convert("RGB")
    logits, cams = run_model(image)

    # The exported model outputs one 7x7 activation map per class
    cam = np.maximum(cams[int(logits.argmax())], 0)
    cam = cam / (cam.max() + 1e-8)

    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    colored = Image.fromarray(np.uint8(_jet(np.asarray(heat) / 255.0) * 255))
    return Image.blend(image, colored, alpha=0.4)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from src.skin.predict import predict_skin

    path = Path(sys.argv[1])
    img = Image.open(path)
    print(predict_skin(img))
    out = Path("reports/figures") / f"gradcam_{path.stem}.png"
    gradcam(img).save(out)
    print("Saved", out)