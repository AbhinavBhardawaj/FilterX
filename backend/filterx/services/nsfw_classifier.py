"""NSFW classifier service.

This module attempts to load a MediaNet TensorFlow model from a stable
path under `backend/models/medianet`. If TensorFlow or the model is not
available it falls back to a lightweight dummy classifier for safe testing.
"""

from pathlib import Path
import base64
import io
import logging
from typing import Literal, Tuple, Any

try:
    import numpy as np
    from PIL import Image
except Exception:
    # Pillow / numpy might not be installed yet; functions will raise later
    np = None
    Image = None

Label = Literal["safe", "nsfw"]
Action = Literal["allow", "blur", "block"]

logger = logging.getLogger(__name__)


# Determine model path relative to this file. This resolves to the `backend/` dir
# when this file is at backend/filterx/services/nsfw_classifier.py
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "medianet"


_TF_AVAILABLE = False
_MEDIANET_MODEL = None
import os
# Allow disabling TensorFlow import in dev environments that have incompatible
# native libraries (e.g. LibreSSL vs OpenSSL issues on macOS). Set
# `FILTERX_DISABLE_TF=1` in the environment to force the dummy classifier.
if os.environ.get("FILTERX_DISABLE_TF"):
    logger.info("TensorFlow import disabled via FILTERX_DISABLE_TF; using dummy classifier")
else:
    try:
        import tensorflow as tf  # type: ignore
        _TF_AVAILABLE = True
    except Exception:
        logger.info("TensorFlow not available; using dummy classifier")


def _load_medianet_model():
    """Load the MediaNet model once at import time.

    Returns the loaded model or None if loading failed.
    """
    if not _TF_AVAILABLE:
        return None

    if not MODEL_PATH.exists():
        logger.warning("MediaNet model path not found: %s", MODEL_PATH)
        return None

    try:
        logger.info("Loading MediaNet model from %s", MODEL_PATH)
        # tf.keras.models.load_model can load SavedModel dirs or HDF5 files
        model = tf.keras.models.load_model(str(MODEL_PATH))
        logger.info("MediaNet model loaded successfully")
        return model
    except Exception as e:
        logger.exception("Failed to load MediaNet model: %s", e)
        return None


# Load once at module import
try:
    _MEDIANET_MODEL = _load_medianet_model()
except Exception:
    _MEDIANET_MODEL = None


class _DummyModel:
    """Very small fallback model for local testing without TensorFlow."""

    def predict_image(self, image_bytes: bytes) -> Tuple[Label, float]:
        # very naive: mark large images with random chance
        import random

        score = random.random()
        if score > 0.75:
            return "nsfw", score
        return "safe", score

    def predict_text(self, text: str) -> Tuple[Label, float]:
        NSFW_KEYWORDS = [
            "xxx",
            "sex",
            "porn",
            "nude",
            "boobs",
            "nsfw",
            "x-rated",
        ]
        text_lower = (text or "").lower()
        hits = [w for w in NSFW_KEYWORDS if w in text_lower]
        if hits:
            confidence = min(0.5 + 0.12 * len(hits), 0.99)
            return "nsfw", confidence
        return "safe", 0.2


_DUMMY = _DummyModel()


def _preprocess_image_for_medianet(image_bytes: bytes, size=(224, 224)) -> Any:
    if Image is None or np is None:
        raise RuntimeError("Pillow and numpy are required for image preprocessing")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def _predict_image_with_medianet(image_bytes: bytes) -> Tuple[Label, float]:
    # If MediaNet model is not available, fallback to dummy
    if _MEDIANET_MODEL is None:
        return _DUMMY.predict_image(image_bytes)

    try:
        processed = _preprocess_image_for_medianet(image_bytes)
    except Exception as e:
        logger.exception("Failed to preprocess image: %s", e)
        return "safe", 0.0

    try:
        preds = _MEDIANET_MODEL.predict(processed)
        # Model output shape may vary. Attempt common cases.
        if hasattr(preds, "shape") and preds.shape[-1] == 1:
            nsfw_prob = float(preds[0][0])
        else:
            # If model outputs two classes [safe_prob, nsfw_prob]
            arr = np.asarray(preds)
            if arr.size == 1:
                nsfw_prob = float(arr.flatten()[0])
            elif arr.shape[-1] >= 2:
                nsfw_prob = float(arr[0, -1])
            else:
                nsfw_prob = float(arr.flatten()[0])

        nsfw_prob = max(0.0, min(1.0, nsfw_prob))
        if nsfw_prob >= 0.7:
            return "nsfw", nsfw_prob
        return "safe", 1.0 - nsfw_prob
    except Exception as e:
        logger.exception("MediaNet prediction failed: %s", e)
        # fallback
        return _DUMMY.predict_image(image_bytes)


def classify_image_base64(b64_string: str) -> dict:
    try:
        image_bytes = base64.b64decode(b64_string.split(",")[-1])
    except Exception:
        return {
            "label": "safe",
            "confidence": 0.0,
            "action": "allow",
            "reason": "Invalid image data",
        }

    label, score = _predict_image_with_medianet(image_bytes)
    action: Action = "allow"
    if label == "nsfw":
        action = "block"

    return {
        "label": label,
        "confidence": float(score),
        "action": action,
        "reason": "MediaNet classifier (image)" if _MEDIANET_MODEL is not None else "Dummy classifier",
    }


def classify_text(text: str) -> dict:
    # Improved keyword-based scoring
    label, score = _DUMMY.predict_text(text)
    action: Action = "allow"
    if label == "nsfw":
        action = "blur"

    reason = "Keyword-based text classifier (dummy)"
    return {
        "label": label,
        "confidence": float(score),
        "action": action,
        "reason": reason,
    }


def classify_url(url: str) -> dict:
    BLOCKLIST = ["porn", "xvideos", "xnxx", "redtube", "xhampster"]
    lower_url = (url or "").lower()
    if any(bad in lower_url for bad in BLOCKLIST):
        return {
            "label": "nsfw",
            "confidence": 0.95,
            "action": "block",
            "reason": "URL matched NSFW blocklist",
        }

    return {
        "label": "safe",
        "confidence": 0.1,
        "action": "allow",
        "reason": "URL not in blocklist",
    }
