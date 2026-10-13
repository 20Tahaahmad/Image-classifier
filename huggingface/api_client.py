import io
import json
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub import ImageClassificationOutputElement
from PIL import Image


# Models run locally with ONNX Runtime, so no Inference Provider credits are used.
# HF_TOKEN in .env is optional and only raises the Hub's download rate limit.
load_dotenv()


DEFAULT_MODEL = "google/vit-base-patch16-224"

# Models available for selection in the UI (id -> display name)
MODELS = {
    "google/vit-base-patch16-224": "ViT Base · Google",
    "facebook/convnext-large-224": "ConvNeXt Large · Meta",
    "microsoft/resnet-50": "ResNet-50 · Microsoft",
    "microsoft/beit-base-patch16-224": "BEiT Base · Microsoft",
    "facebook/deit-base-distilled-patch16-224": "DeiT Base · Meta",
}


def _onnx_repo(model):
    # ONNX exports of the original checkpoints
    return "Xenova/" + model.split("/")[1]


# The quantized ResNet-50 export gives wrong predictions, so use full precision
FULL_PRECISION = {"microsoft/resnet-50"}


@lru_cache(maxsize=None)
def _load(model):
    """Download (once, then cached on disk) and load a model."""

    repo = _onnx_repo(model)
    weights = "onnx/model.onnx" if model in FULL_PRECISION else "onnx/model_quantized.onnx"

    session = ort.InferenceSession(
        hf_hub_download(repo, weights),
        providers=["CPUExecutionProvider"]
    )

    with open(hf_hub_download(repo, "preprocessor_config.json")) as f:
        preprocessor = json.load(f)

    with open(hf_hub_download(repo, "config.json")) as f:
        id2label = json.load(f)["id2label"]

    return session, preprocessor, id2label


def _preprocess(image, config):
    """Resize, crop and normalize like the model's Hugging Face image processor."""

    resample = Image.Resampling(config.get("resample", 2))
    size = config["size"]

    if "shortest_edge" in size:
        # ConvNeXt / ResNet: resize shortest edge to 224 / crop_pct, then center crop 224
        crop = size["shortest_edge"]
        edge = int(crop / config.get("crop_pct", 1.0))
        scale = edge / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), resample)
        crop_w = crop_h = crop
    else:
        image = image.resize((size["width"], size["height"]), resample)
        crop_w, crop_h = image.size

        if config.get("do_center_crop"):
            crop_w, crop_h = config["crop_size"]["width"], config["crop_size"]["height"]

    left = (image.width - crop_w) // 2
    top = (image.height - crop_h) // 2
    image = image.crop((left, top, left + crop_w, top + crop_h))

    pixels = np.asarray(image, dtype=np.float32) * config.get("rescale_factor", 1 / 255)
    pixels = (pixels - config["image_mean"]) / config["image_std"]

    return pixels.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def classify_image(image, model=DEFAULT_MODEL, top_k=5):
    """Classify an image given as a file path or raw bytes."""

    session, preprocessor, id2label = _load(model)

    if isinstance(image, bytes):
        image = io.BytesIO(image)

    with Image.open(image) as img:
        pixels = _preprocess(img.convert("RGB"), preprocessor)

    logits = session.run(None, {session.get_inputs()[0].name: pixels})[0][0]

    scores = np.exp(logits - logits.max())
    scores /= scores.sum()

    return [
        ImageClassificationOutputElement(label=id2label[str(i)], score=float(scores[i]))
        for i in np.argsort(scores)[::-1][:top_k]
    ]
