"""Checksum-pinned model downloads. Run with python models.py."""
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent / "models"
V1_SOURCE = ("https://huggingface.co/e96031413/mobilevlm-app-models/resolve/main/"
             "2026.07.25.1/minivision-ai_MiniFASNetV2/")
MODEL_SOURCES = {
    "minifas_v1se.onnx": (
        V1_SOURCE + "minifas_v1se.onnx",
        "2460ec78f0df1dbc14d7bc7a4072361c26147291c105fe841a8eb33066bcb6e7",
    ),
    "minifas_v1se.onnx.data": (
        V1_SOURCE + "minifas_v1se.onnx.data",
        "c400781ff6aef1d476b2094ac2ac94b0323fa835a62dd63220d060b8cdce29bd",
    ),
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    "minifasnet_v2.onnx": (
        "https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx/resolve/main/minifasnet_v2.onnx",
        "d7b3cd9ba8a7ceb13baa8c4720902e27ca3112eff52f926c08804af6b6eecc7b",
    ),
}


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_models():
    return all((ROOT / name).is_file() and checksum(ROOT / name) == digest
               for name, (_, digest) in MODEL_SOURCES.items())


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    for name, (url, digest) in MODEL_SOURCES.items():
        target = ROOT / name
        if target.is_file() and checksum(target) == digest:
            print(f"Verified {name}")
            continue
        partial = target.with_suffix(".part")
        try:
            with urlopen(url, timeout=30) as source, partial.open("wb") as output:
                for block in iter(lambda: source.read(256 * 1024), b""):
                    output.write(block)
            if checksum(partial) != digest:
                raise ValueError(f"Unexpected SHA-256 for {name}")
            os.replace(partial, target)
            print(f"Downloaded and verified {name}")
        finally:
            partial.unlink(missing_ok=True)
