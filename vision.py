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


DEFAULT_GUIDE = {
    "left": 0.27,
    "top": 0.16,
    "width": 0.46,
    "height": 0.65,
}
OVERLAP_THRESHOLD = 0.50


def face_guide_overlap(face_box, guide_box):
    """
    Computes overlap metrics between a face bounding box and the guide region.
    face_box: (x, y, w, h) in pixel coordinates.
    guide_box: (gx1, gy1, gx2, gy2) in pixel coordinates.

    Returns:
        overlap_ratio: float in [0.0, 1.0], fraction of face_box area inside guide_box.
        center_inside: bool, True if the geometric center (cx, cy) is inside guide_box.
    """
    fx, fy, fw, fh = face_box[:4]
    if fw <= 0 or fh <= 0:
        return 0.0, False

    gx1, gy1, gx2, gy2 = guide_box
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0

    center_inside = (gx1 <= cx <= gx2) and (gy1 <= cy <= gy2)

    ix1 = max(fx, gx1)
    iy1 = max(fy, gy1)
    ix2 = min(fx + fw, gx2)
    iy2 = min(fy + fh, gy2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter_area = iw * ih
    face_area = fw * fh

    overlap_ratio = inter_area / face_area if face_area > 0 else 0.0
    return overlap_ratio, center_inside


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
    def __init__(self, live_min=.85, spoof_max=.25, check_enroll_pad=True, guide=None, overlap_threshold=OVERLAP_THRESHOLD):
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
        self.guide = dict(guide) if guide is not None else DEFAULT_GUIDE.copy()
        self.overlap_threshold = float(overlap_threshold)
        self.detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
        self.recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
        for session, input_name, _ in self.pad_models:
            self.pad_score(session, input_name, np.zeros((80, 80, 3), dtype=np.uint8))

    def faces(self, image, previous_face=None):
        height, width = image.shape[:2]
        small = cv2.resize(image, (320, 240))
        _, raw_faces = self.detector.detect(small)

        if raw_faces is None or len(raw_faces) == 0:
            raise CaptureError("center_face_in_guide")

        # Map detected face coordinates back to original image dimensions
        scaled_faces = []
        for rf in raw_faces:
            f = rf.copy()
            f[:14:2] *= width / 320.0
            f[1:14:2] *= height / 240.0
            scaled_faces.append(f)

        # Active face region corresponding to the visible guide
        gx1 = self.guide["left"] * width
        gy1 = self.guide["top"] * height
        gx2 = (self.guide["left"] + self.guide["width"]) * width
        gy2 = (self.guide["top"] + self.guide["height"]) * height
        guide_box = (gx1, gy1, gx2, gy2)

        # Overlap rule:
        # 1. Geometric center (cx, cy) must lie within the guide rectangle.
        # 2. At least OVERLAP_THRESHOLD (50%) of face area must overlap the guide.
        # Background people, passersby, or tiny boundary intersections are ignored.
        eligible_faces = []
        for face_candidate in scaled_faces:
            overlap_ratio, center_inside = face_guide_overlap(face_candidate[:4], guide_box)
            if center_inside and overlap_ratio >= self.overlap_threshold:
                eligible_faces.append(face_candidate)

        if len(eligible_faces) == 0:
            # Faces detected in frame, but none centered substantially inside the guide
            raise CaptureError("center_face_in_guide")

        if len(eligible_faces) > 1:
            # Multiple people positioned inside the guide
            raise CaptureError("one_face_required")

        face = eligible_faces[0]

        # Burst consistency check: ensure face does not switch or jump between people
        if previous_face is not None:
            prev_cx = previous_face[0] + previous_face[2] / 2.0
            prev_cy = previous_face[1] + previous_face[3] / 2.0
            curr_cx = face[0] + face[2] / 2.0
            curr_cy = face[1] + face[3] / 2.0
            if math.hypot(curr_cx - prev_cx, curr_cy - prev_cy) > 0.35 * width:
                raise CaptureError("center_face_in_guide")

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
            face_results = []
            prev_face = None
            for image in samples:
                face, qual = self.faces(image, previous_face=prev_face)
                face_results.append((face, qual))
                prev_face = face
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
            prev_face = None
            for i, frame in enumerate(frames):
                image = decode(frame)
                face, quality = self.faces(image, previous_face=prev_face)
                samples.append((image, face))
                prev_face = face
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
            frame_embeddings = []
            for idx, (image, face) in enumerate(samples):
                emb = self.embedding(image, face)
                frame_embeddings.append(emb)
                val = float(emb @ enrolled)
                if not math.isfinite(val):
                    return Evidence("UNCERTAIN", score, reason="identity_uncertain", diagnostics=diag_summary)
                frame_matches.append(val)
                frame_diagnostics[idx]["identity_score"] = round(val, 4)

            # Ensure identity did not switch across frames in the burst
            if min(float(frame_embeddings[0] @ emb) for emb in frame_embeddings[1:]) < .50:
                return Evidence("UNCERTAIN", score, reason="identity_uncertain", diagnostics=diag_summary)

            match = min(frame_matches)
            diag_summary["identity_min"] = round(match, 4)
            diag_summary["identity_scores"] = [round(m, 4) for m in frame_matches]
            return Evidence("LIVE", score, match, reason="verified", diagnostics=diag_summary)
        except CaptureError as exc:
            return Evidence("UNCERTAIN", reason=str(exc))
        finally:
            self.lock.release()
