"""Local CPU inference. Models are loaded and warmed before accepting scans."""
import base64
import io
import math
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from models import ROOT, check_models


class CaptureError(Exception):
    pass


@dataclass
class Evidence:
    pad: str
    live_score: float | None = None
    identity_score: float | None = None
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)


def decode(frame):
    try:
        payload = base64.b64decode(frame, validate=True)
        with Image.open(io.BytesIO(payload), formats=("JPEG",)) as image:
            if not (160 <= image.width <= 640 and 120 <= image.height <= 480):
                raise CaptureError("invalid_frame_size")
            return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    except CaptureError:
        raise
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise CaptureError("invalid_frame") from exc


def crop_with_context(image, face, margin):
    x, y, w, h = face[:4]
    height, width = image.shape[:2]
    if w <= 0 or h <= 0:
        raise CaptureError("invalid_face_crop")
    scale = min(margin, (width - 1) / w, (height - 1) / h)
    crop_width, crop_height = int(w * scale), int(h * scale)
    if crop_width <= 0 or crop_height <= 0:
        raise CaptureError("invalid_face_crop")
    left = int(np.clip(x + (w - crop_width) / 2, 0, width - crop_width))
    top = int(np.clip(y + (h - crop_height) / 2, 0, height - crop_height))
    return cv2.resize(image[top:top + crop_height, left:left + crop_width], (80, 80))


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size != 128:
        raise CaptureError("invalid_embedding_dimensions")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise CaptureError("invalid_embedding")
    return vector / norm


def validate_template(template):
    """Ensure decrypted template bytes form a valid, finite 128-d unit vector."""
    if not isinstance(template, (bytes, bytearray)) or len(template) != 512:
        raise ValueError("invalid_template_length")
    vector = np.frombuffer(template, dtype=np.float32)
    if vector.size != 128 or not np.isfinite(vector).all():
        raise ValueError("invalid_template_values")
    norm = float(np.linalg.norm(vector))
    if not (0.98 <= norm <= 1.02):
        raise ValueError("invalid_template_norm")
    return vector


class Vision:
    def __init__(self, live_min=.85, spoof_max=.25, check_enroll_pad=True):
        if not 0 < spoof_max < live_min < 1:
            raise ValueError("Require 0 < spoof_max < live_min < 1")
        if not check_models():
            raise RuntimeError("Missing or invalid models. Run python models.py")
        cv2.setNumThreads(2)
        self.detector = cv2.FaceDetectorYN.create(
            str(ROOT / "face_detection_yunet_2023mar.onnx"), "", (320, 240), .75, .3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(ROOT / "face_recognition_sface_2021dec.onnx"), "")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.pad_models = []
        for name, margin in (("minifasnet_v2.onnx", 2.7), ("minifas_v1se.onnx", 4.0)):
            session = ort.InferenceSession(str(ROOT / name), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            self.pad_models.append((session, session.get_inputs()[0].name, margin))
        self.lock = threading.Lock()
        self.live_min, self.spoof_max = live_min, spoof_max
        self.check_enroll_pad = check_enroll_pad and os.environ.get("LIVEGUARD_DISABLE_ENROLL_PAD") != "1"
        self.detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        self.recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
        for session, input_name, _ in self.pad_models:
            self.pad_score(session, input_name, np.zeros((80, 80, 3), dtype=np.uint8))

    def faces(self, image):
        height, width = image.shape[:2]
        small = cv2.resize(image, (320, 240))
        _, faces = self.detector.detect(small)
        if faces is None or len(faces) != 1:
            raise CaptureError("one_face_required")
        face = faces[0].copy()
        face[:14:2] *= width / 320
        face[1:14:2] *= height / 240
        x, y, w, h = face[:4]
        if min(w, h) < 85 or x < 3 or y < 3 or x + w >= width - 3 or y + h >= height - 3:
            raise CaptureError("center_face_in_guide")
        gray = cv2.cvtColor(image[int(y):int(y+h), int(x):int(x+w)], cv2.COLOR_BGR2GRAY)
        light = float(gray.mean())
        focus = float(cv2.Laplacian(cv2.resize(gray, (112, 112)), cv2.CV_32F).var())
        if not 45 <= light <= 225 or focus < 25:
            raise CaptureError("improve_light_or_focus")
        return face, {"light": light, "focus": focus, "box": [float(x), float(y), float(w), float(h)]}

    @staticmethod
    def pad_score(session, input_name, crop):
        # Upstream numpy ToTensor casts to float and preserves the 0–255 range.
        sample = crop.astype(np.float32).transpose(2, 0, 1)[None]
        logits = session.run(None, {input_name: np.ascontiguousarray(sample)})[0].reshape(-1)
        if logits.size != 3 or not np.isfinite(logits).all():
            raise RuntimeError("Unexpected PAD output")
        exp = np.exp(logits - logits.max())
        # Original upstream MiniFASNet uses class 1 for live.
        return float(exp[1] / exp.sum())

    def embedding(self, image, face):
        return normalized(self.recognizer.feature(self.recognizer.alignCrop(image, face)))

    def enroll(self, frames):
        if not self.lock.acquire(blocking=False):
            raise CaptureError("camera_pipeline_busy")
        try:
            samples = [decode(frame) for frame in frames]
            face_results = [self.faces(image) for image in samples]
            faces = [res[0] for res in face_results]

            # Live Presentation Attack Detection on enrollment frames
            if self.check_enroll_pad:
                scores = []
                for image, face in zip(samples, faces):
                    scores.append([self.pad_score(session, input_name, crop_with_context(image, face, margin))
                                   for session, input_name, margin in self.pad_models])
                if sum(max(pair) <= self.spoof_max for pair in scores) >= 2 and np.max(scores) < self.live_min:
                    raise CaptureError("enrollment_presentation_attack")
                if np.min(scores) < self.live_min:
                    raise CaptureError("enrollment_liveness_uncertain")

            vectors = [self.embedding(image, face) for image, face in zip(samples, faces)]
            if min(float(vectors[0] @ vector) for vector in vectors[1:]) < .50:
                raise CaptureError("enrollment_face_changed")
            return normalized(np.mean(vectors, axis=0)).tobytes()
        finally:
            self.lock.release()

    def verify(self, frames, template):
        if not self.lock.acquire(blocking=False):
            return Evidence("UNCERTAIN", reason="camera_pipeline_busy")
        try:
            samples = []
            scores = []
            frame_diagnostics = []
            for i, frame in enumerate(frames):
                image = decode(frame)
                face, quality = self.faces(image)
                samples.append((image, face))
                pad_pair = [self.pad_score(session, input_name, crop_with_context(image, face, margin))
                            for session, input_name, margin in self.pad_models]
                scores.append(pad_pair)
                frame_diagnostics.append({
                    "frame": i + 1,
                    "light": round(quality["light"], 1),
                    "focus": round(quality["focus"], 1),
                    "pad_v2": round(pad_pair[0], 4),
                    "pad_v1se": round(pad_pair[1], 4),
                })

            score = float(np.median(scores))
            min_score = float(np.min(scores))
            max_score = float(np.max(scores))

            diag_summary = {
                "frames": frame_diagnostics,
                "pad_median": round(score, 4),
                "pad_min": round(min_score, 4),
                "pad_max": round(max_score, 4),
            }

            if sum(max(pair) <= self.spoof_max for pair in scores) >= 2 and max_score < self.live_min:
                return Evidence("SPOOF", score, reason="presentation_attack", diagnostics=diag_summary)
            if min_score < self.live_min:
                return Evidence("UNCERTAIN", score, reason="liveness_uncertain", diagnostics=diag_summary)
            if template is None:
                return Evidence("LIVE", score, reason="identity_not_enrolled", diagnostics=diag_summary)

            # Validate enrolled template to prevent malformed/NaN embeddings
            try:
                enrolled = validate_template(template)
            except ValueError:
                return Evidence("UNCERTAIN", score, reason="template_corrupt", diagnostics=diag_summary)

            # Match each frame so mixed identities cannot be averaged into an allow
            frame_matches = []
            for idx, (image, face) in enumerate(samples):
                emb = self.embedding(image, face)
                val = float(emb @ enrolled)
                if not math.isfinite(val):
                    return Evidence("UNCERTAIN", score, reason="identity_uncertain", diagnostics=diag_summary)
                frame_matches.append(val)
                frame_diagnostics[idx]["identity_score"] = round(val, 4)

            match = min(frame_matches)
            diag_summary["identity_min"] = round(match, 4)
            diag_summary["identity_scores"] = [round(m, 4) for m in frame_matches]
            return Evidence("LIVE", score, match, reason="verified", diagnostics=diag_summary)
        except CaptureError as exc:
            return Evidence("UNCERTAIN", reason=str(exc))
        finally:
            self.lock.release()
