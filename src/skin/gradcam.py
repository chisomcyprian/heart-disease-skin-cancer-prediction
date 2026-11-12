import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from src.skin.predict import _tf, load_model


def gradcam(image: Image.Image) -> Image.Image:
    """Return the image overlaid with a heatmap of where the model looked."""
    model = load_model()
    image = image.convert("RGB")
    x = _tf(image).unsqueeze(0)

    # Capture the output of the last convolutional block
    store = {}
    hook = model.features[-1].register_forward_hook(
        lambda module, inp, out: store.update(act=out)
    )
    try:
        out = model(x)
    finally:
        hook.remove()

    # How much does each feature map push the winning class?
    score = out[0, out.argmax()]
    grads = torch.autograd.grad(score, store["act"])[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * store["act"]).sum(dim=1))[0]
    cam = (cam / (cam.max() + 1e-8)).detach().numpy()

    # Stretch the 7x7 map to image size, colour it, blend it
    heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
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