"""
webcam.py — Real-Time Face Recognition Video Loop
==================================================

Responsibility:
    Implements the complete real-time face recognition pipeline using the
    laptop webcam as input. Orchestrates:

        Capture frame → Detect faces → Crop each face → Recognise →
        Draw overlay (bounding box + name + confidence) → Display → Repeat

    Three components with clear responsibilities:

    1. FaceDetector (ABC)
       Unified interface for face detection. Abstracts the specific
       detector implementation from the video loop.

    2. YuNetDetector (Concrete)
       Uses cv2.FaceDetectorYN — the only high-quality face detector
       available in OpenCV 5. The ONNX model is auto-downloaded from
       the OpenCV Zoo on first use and cached locally.

    3. WebcamRecognition
       The main video loop. Depends only on the FaceDetector ABC and
       the Recognizer — not on any concrete implementation. (DIP)

Note on OpenCV 5:
    OpenCV 5 removed CascadeClassifier and HOGDescriptor. The only
    supported face detector is cv2.FaceDetectorYN (deep-learning based).
    Its model is downloaded automatically on first run (~1 MB).

Quit:
    Press 'Q' at any time to exit the webcam loop gracefully.

Usage:
    from src.recognition.webcam import WebcamRecognition, create_detector
    from src.recognition.recognizer import Recognizer

    detector = create_detector()           # auto-downloads YuNet if needed
    webcam   = WebcamRecognition(recognizer=recognizer, detector=detector)
    webcam.run()
"""

from __future__ import annotations

import logging
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.config.settings import get_settings
from src.recognition.recognizer import RecognitionResult, Recognizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Overlay constants
# ---------------------------------------------------------------------------

_COLOR_KNOWN:   Tuple[int, int, int] = (0, 220, 0)    # Green (BGR)
_COLOR_UNKNOWN: Tuple[int, int, int] = (0, 0, 220)    # Red   (BGR)
_COLOR_WHITE:   Tuple[int, int, int] = (255, 255, 255)
_COLOR_FPS:     Tuple[int, int, int] = (0, 200, 255)  # Amber

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_BOX_THICK  = 2
_TEXT_SCALE = 0.6
_TEXT_THICK = 1

# YuNet ONNX model auto-download source (OpenCV Zoo, official)
_YUNET_URL:        str = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_CACHE_NAME: str = "face_detection_yunet_2023mar.onnx"


# ---------------------------------------------------------------------------
# FaceDetector — Abstract Base Class
# ---------------------------------------------------------------------------

class FaceDetector(ABC):
    """
    Abstract interface for face detection.

    Defines the single contract any face detector must fulfil:
    given a BGR frame, return a list of (x, y, w, h) bounding boxes.

    WebcamRecognition depends only on this interface — swapping the
    underlying detector requires zero changes in the video loop.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all faces in a BGR frame.

        Args:
            frame: BGR image from cv2.VideoCapture, shape (H, W, 3).

        Returns:
            List[Tuple[int,int,int,int]]: (x, y, w, h) bounding boxes.
                Empty if no faces found. Boxes below MIN_FACE_SIZE excluded.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name for logging."""
        ...


# ---------------------------------------------------------------------------
# YuNetDetector
# ---------------------------------------------------------------------------

class YuNetDetector(FaceDetector):
    """
    Face detector using OpenCV's YuNet deep-learning model.

    YuNet (Yu et al., 2022) is a lightweight, high-accuracy face detector.
    It is the only face detector available in OpenCV 5 out-of-the-box
    (CascadeClassifier was removed in OpenCV 5).

    The ONNX model is loaded from:
        1. Settings.YUNET_MODEL_PATH  (if configured by the user)
        2. Local cache in outputs/ directory (if previously downloaded)
        3. Auto-downloaded from OpenCV Zoo on first use (~1 MB)

    YuNet bounding box format:
        [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm,
         x_lcm, y_lcm, confidence_score]
    We use only x, y, w, h (indices 0–3).
    """

    def __init__(
        self,
        model_path:     str,
        conf_threshold: float,
        nms_threshold:  float,
        top_k:          int = 5000,
    ) -> None:
        """
        Initialise YuNetDetector with a validated ONNX model path.

        Args:
            model_path:     Path to a valid YuNet ONNX model file.
            conf_threshold: Minimum detection confidence [0, 1].
            nms_threshold:  NMS overlap threshold [0, 1].
            top_k:          Max detections before NMS.
        """
        self._detector = cv2.FaceDetectorYN.create(
            model           = model_path,
            config          = "",
            input_size      = (320, 320),   # updated per-frame in detect()
            score_threshold = conf_threshold,
            nms_threshold   = nms_threshold,
            top_k           = top_k,
        )
        self._last_size: Tuple[int, int] = (0, 0)
        self._min_face = get_settings().MIN_FACE_SIZE
        logger.info(f"YuNetDetector ready (model: '{Path(model_path).name}')")

    @property
    def name(self) -> str:
        return "YuNet"

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Run YuNet inference on one BGR frame.

        Updates the input_size whenever the frame resolution changes.
        This is required by YuNet's internal NMS logic.

        Args:
            frame: BGR frame, shape (H, W, 3).

        Returns:
            List[Tuple[int,int,int,int]]: Validated (x, y, w, h) boxes.
        """
        h, w = frame.shape[:2]
        if (w, h) != self._last_size:
            self._detector.setInputSize((w, h))
            self._last_size = (w, h)

        _, detections = self._detector.detect(frame)

        if detections is None:
            return []

        boxes = []
        for det in detections:
            x, y, bw, bh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            if bw >= self._min_face and bh >= self._min_face:
                boxes.append((x, y, bw, bh))
        return boxes


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_detector() -> FaceDetector:
    """
    Create the best available face detector, auto-downloading if needed.

    Priority:
        1. Configured path in Settings.YUNET_MODEL_PATH
        2. Cached model in outputs/ directory
        3. Auto-download from OpenCV Zoo → cache → initialise

    Returns:
        YuNetDetector: A ready-to-use detector instance.

    Raises:
        RuntimeError: If YuNet cannot be initialised and no fallback exists.
    """
    cfg = get_settings()

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "cv2.FaceDetectorYN is not available in this OpenCV build. "
            "Please upgrade: pip install opencv-python>=4.5.4"
        )

    # ── Priority 1: Configured model path ───────────────────────────────
    if cfg.YUNET_MODEL_PATH and Path(cfg.YUNET_MODEL_PATH).exists():
        try:
            return YuNetDetector(
                cfg.YUNET_MODEL_PATH,
                cfg.FACE_CONFIDENCE_THRESHOLD,
                cfg.FACE_NMS_THRESHOLD,
            )
        except Exception as exc:
            logger.warning(f"Configured YuNet model failed: {exc}")

    # ── Priority 2 & 3: Cache or download ───────────────────────────────
    yunet_cache = cfg.OUTPUTS_DIR / _YUNET_CACHE_NAME

    if not yunet_cache.exists():
        logger.info(
            f"YuNet ONNX model not found locally. "
            f"Downloading from OpenCV Zoo → {yunet_cache} ..."
        )
        yunet_cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(_YUNET_URL, yunet_cache)
            logger.info("YuNet model downloaded and cached successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download YuNet model: {exc}\n"
                f"Please manually download the ONNX file from:\n  {_YUNET_URL}\n"
                f"and place it at:\n  {yunet_cache}\n"
                f"or set YUNET_MODEL_PATH in src/config/settings.py"
            ) from exc

    return YuNetDetector(
        str(yunet_cache),
        cfg.FACE_CONFIDENCE_THRESHOLD,
        cfg.FACE_NMS_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# WebcamRecognition — the main video loop
# ---------------------------------------------------------------------------

class WebcamRecognition:
    """
    Real-time face recognition using the laptop webcam.

    Responsibility:
        Manage the capture-detect-recognise-display loop. Delegates all
        detection to the injected FaceDetector and all recognition to the
        injected Recognizer — the loop itself owns no ML logic.

    Loop:
        1. Capture BGR frame from webcam
        2. Detect faces → List[(x, y, w, h)]
        3. Crop each face with 15% padding
        4. recognizer.recognize_batch(crops)
        5. Draw 🟩/🟥 bounding box + name + confidence per face
        6. Draw FPS counter (rolling 30-frame average)
        7. Display in OpenCV window
        8. Exit on 'Q' keypress

    Usage:
        detector = create_detector()
        cam = WebcamRecognition(recognizer=recognizer, detector=detector)
        cam.run()
    """

    _PAD_FACTOR: float = 0.15  # 15% padding around detected face box
    _FPS_WINDOW: int   = 30    # Rolling average window size (frames)

    def __init__(
        self,
        recognizer: Recognizer,
        detector:   Optional[FaceDetector] = None,
    ) -> None:
        """
        Initialise the WebcamRecognition session.

        Args:
            recognizer: Fully initialised Recognizer instance.
            detector:   FaceDetector to use. If None, create_detector()
                        is called to select/download the best available.
        """
        self._cfg        = get_settings()
        self._recognizer = recognizer
        self._detector   = detector or create_detector()
        self._frame_times: Deque[float] = deque(maxlen=self._FPS_WINDOW)

        logger.info(
            f"WebcamRecognition ready — "
            f"detector: {self._detector.name}, "
            f"camera index: {self._cfg.WEBCAM_INDEX}"
        )

    def run(self) -> None:
        """
        Start the real-time recognition loop.

        Opens the webcam and processes frames until the user presses 'Q',
        the feed is lost, or a KeyboardInterrupt (Ctrl+C) is received.

        Raises:
            RuntimeError: If the webcam cannot be opened.
        """
        cap = self._open_camera()
        logger.info("Webcam open. Press 'Q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed — camera disconnected?")
                    break

                t0        = time.perf_counter()
                annotated = self._process_frame(frame)
                self._frame_times.append(time.perf_counter() - t0)

                self._draw_fps(annotated, self._compute_fps())
                cv2.imshow("Siamese Face Recognition  [Q to quit]", annotated)

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    logger.info("Quit key pressed.")
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released. Session ended.")

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect → crop → recognise → annotate one webcam frame."""
        annotated = frame.copy()
        boxes     = self._detector.detect(frame)

        if not boxes:
            self._draw_no_face_hint(annotated)
            return annotated

        valid_boxes: List[Tuple[int, int, int, int]] = []
        face_crops:  List[np.ndarray]                = []

        for box in boxes:
            crop = self._crop_face(frame, box)
            if crop is not None:
                valid_boxes.append(box)
                face_crops.append(crop)

        if not face_crops:
            return annotated

        results = self._recognizer.recognize_batch(face_crops)
        for box, result in zip(valid_boxes, results):
            self._draw_face_overlay(annotated, box, result)

        return annotated

    def _crop_face(
        self,
        frame: np.ndarray,
        box:   Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Safely crop a padded face region, clamped to frame bounds.

        Adds PAD_FACTOR padding on all sides. Returns None if the
        resulting crop would be empty or entirely outside the frame.
        """
        x, y, w, h = box
        H, W = frame.shape[:2]

        x1 = max(0, x - int(w * self._PAD_FACTOR))
        y1 = max(0, y - int(h * self._PAD_FACTOR))
        x2 = min(W, x + w + int(w * self._PAD_FACTOR))
        y2 = min(H, y + h + int(h * self._PAD_FACTOR))

        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_face_overlay(
        self,
        frame:  np.ndarray,
        box:    Tuple[int, int, int, int],
        result: RecognitionResult,
    ) -> None:
        """Draw coloured bounding box, name label, and distance value."""
        x, y, w, h = box
        color = _COLOR_KNOWN if result.is_known else _COLOR_UNKNOWN

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, _BOX_THICK)

        label = result.display_name + (f"  {result.confidence:.0%}" if result.is_known else "")
        (lw, lh), baseline = cv2.getTextSize(label, _FONT, _TEXT_SCALE, _TEXT_THICK)
        label_y = max(y - 4, lh + baseline)

        cv2.rectangle(
            frame,
            (x, label_y - lh - baseline),
            (x + lw + 4, label_y + baseline),
            color, cv2.FILLED,
        )
        cv2.putText(
            frame, label, (x + 2, label_y - baseline // 2),
            _FONT, _TEXT_SCALE, _COLOR_WHITE, _TEXT_THICK, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"d={result.distance:.3f}", (x, y + h + 16),
            _FONT, 0.45, color, 1, cv2.LINE_AA,
        )

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw rolling-average FPS counter in top-left corner."""
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 28),
            _FONT, 0.75, _COLOR_FPS, 2, cv2.LINE_AA,
        )

    def _draw_no_face_hint(self, frame: np.ndarray) -> None:
        """Draw centred 'No face detected' message."""
        H, W = frame.shape[:2]
        label = "No face detected"
        (lw, _), _ = cv2.getTextSize(label, _FONT, 0.7, 2)
        cv2.putText(
            frame, label, ((W - lw) // 2, H - 20),
            _FONT, 0.7, _COLOR_UNKNOWN, 2, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def _open_camera(self) -> cv2.VideoCapture:
        """Open and configure the webcam. Raises RuntimeError if unavailable."""
        cap = cv2.VideoCapture(self._cfg.WEBCAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._cfg.WEBCAM_INDEX}."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.WEBCAM_HEIGHT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera resolution: {w}×{h}")
        return cap

    def _compute_fps(self) -> float:
        """Rolling-average FPS over the last _FPS_WINDOW frames."""
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0 else 0.0

    @property
    def detector(self) -> FaceDetector:
        """The face detector in use."""
        return self._detector
