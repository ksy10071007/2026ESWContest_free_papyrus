from __future__ import annotations

import argparse
import base64
import os
from typing import Any, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request

from model_loader import initialize_model_loader, predict_eye_disease


app = Flask(__name__)
MODEL: Optional[Any] = None


def decode_base64_image(base64_image_data: str):
    if not base64_image_data:
        return None, "empty image payload"
    try:
        encoded = base64_image_data.split(",", 1)[1] if "," in base64_image_data else base64_image_data
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, "failed to decode image"
        return img_bgr, None
    except Exception as exc:
        return None, str(exc)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": MODEL.device if MODEL else None}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    image_data = data.get("image")
    img_bgr, err = decode_base64_image(image_data)
    if err:
        return jsonify({"status": "error", "message": err}), 400

    result = predict_eye_disease(img_bgr)
    return jsonify(result), 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Eye Diagnosis API server")
    parser.add_argument("--device", choices=["jetson", "rpi"], default=None)
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "5000")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL = initialize_model_loader(device=args.device)
    app.run(host=args.host, port=args.port)
