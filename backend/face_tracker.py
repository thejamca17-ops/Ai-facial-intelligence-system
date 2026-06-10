"""
Multi-Face Tracker - Track multiple faces with unique stable IDs
Uses IoU + centroid hybrid matching with Hungarian assignment for optimal matching.
Each track stores recognition state to prevent name switching.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from collections import OrderedDict


class FaceTrack:
    """Represents a single tracked face with stable identity"""

    def __init__(self, face_id: int, bbox: Tuple[int, int, int, int], timestamp: float):
        self.id = face_id
        self.bbox = bbox  # (x, y, w, h)
        self.centroid = self._calculate_centroid(bbox)
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.disappeared_count = 0

        # --- Recognition state (managed by processor) ---
        self.name = "Unknown"
        self.recognition_confidence = 0.0
        self.registered_gender = "Unknown"
        self.needs_recognition = True       # True until first successful recognition
        self.recognition_locked = False     # True once high-confidence name assigned
        self.last_recognition_attempt = 0.0 # Cooldown timer
        self.recognition_attempts = 0       # How many times recognition was tried

        # --- Per-track blink / EAR state (prevents global state mixing) ---
        self.blink_counter = 0          # Cumulative blink count for this track
        self.prev_ear = 0.3             # Previous eye aspect ratio
        self.was_blinking = False       # Tracks blink transitions
        self.blink_total = 0            # Total blinks logged for fatigue

        # --- Per-track DeepFace state (age, gender, emotion) ---
        self.deepface_result = None         # Latest DeepFace result dict
        self.is_processing_deepface = False # Prevents double-launching for this track
        self.last_deepface_update = 0.0     # Cooldown timer for DeepFace per track
        
        # Smoothing buffers for demographics
        self.age_history = []
        self.gender_history = []
        self.emotion_history = []
        self.history_limit = 15             # Keep last 15 predictions for smoothing

        # --- Per-track attribute data (populated each frame from mesh) ---
        self.head_pose = {}
        self.gaze = {}
        self.blink = {}
        self.attention = {}
        self.fatigue = {}
        self.stress = {}
        self.anti_spoof = {}

        # Anti-spoofing data
        self.anti_spoof_status = "Checking..."
        self.anti_spoof_confidence = 0.0
        self.is_live = None

        # Duration tracking
        self.duration_seconds = 0.0

        # Velocity estimation for prediction
        self._prev_centroid = self.centroid
        self.velocity = (0.0, 0.0)


    def _calculate_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    def update_position(self, new_bbox: Tuple[int, int, int, int], timestamp: float):
        """Update face position and estimate velocity"""
        self._prev_centroid = self.centroid
        self.bbox = new_bbox
        self.centroid = self._calculate_centroid(new_bbox)
        dt = max(timestamp - self.last_seen, 0.001)
        self.velocity = (
            (self.centroid[0] - self._prev_centroid[0]) / dt,
            (self.centroid[1] - self._prev_centroid[1]) / dt
        )
        self.last_seen = timestamp
        self.disappeared_count = 0
        self.duration_seconds = timestamp - self.first_seen

    def predicted_centroid(self, dt: float) -> Tuple[float, float]:
        """Predict where the face will be based on velocity"""
        return (
            self.centroid[0] + self.velocity[0] * dt,
            self.centroid[1] + self.velocity[1] * dt
        )

    def mark_disappeared(self):
        """Increment disappeared counter"""
        self.disappeared_count += 1

    def set_recognition(self, name: str, confidence: float, gender: str = "Unknown"):
        """Store recognition result for this track"""
        self.recognition_attempts += 1
        
        if name != "Unknown":
            # Got a real match — update if better confidence or not locked
            if not self.recognition_locked or confidence > self.recognition_confidence:
                self.name = name
                self.recognition_confidence = confidence
                self.registered_gender = gender
                self.needs_recognition = False
                # Lock if confidence is high enough
                if confidence >= 60.0:
                    self.recognition_locked = True
                else:
                    # Medium confidence — allow one more retry to confirm
                    if self.recognition_attempts < 3:
                        self.needs_recognition = True
            
            # If recognized, use registered gender to seed history for stability
            if gender != "Unknown":
                self.gender_history.append(gender)
                if len(self.gender_history) > self.history_limit:
                    self.gender_history.pop(0)

        else:
            # Unknown result — retry up to 3 times, then settle as Unknown
            if self.recognition_attempts < 3:
                self.needs_recognition = True  # Allow retry
            else:
                # Exhausted retries — settle as definitively Unknown
                self.needs_recognition = False
                self.name = "Unknown"
                self.recognition_confidence = confidence

    def update_demographics(self, age: Optional[int], gender: Optional[str], emotion: Optional[str]):
        """Update demographic history for smoothing"""
        if age is not None:
            self.age_history.append(age)
            if len(self.age_history) > self.history_limit:
                self.age_history.pop(0)
        
        if gender is not None:
            self.gender_history.append(gender)
            if len(self.gender_history) > self.history_limit:
                self.gender_history.pop(0)

        if emotion is not None:
            self.emotion_history.append(emotion)
            if len(self.emotion_history) > self.history_limit:
                self.emotion_history.pop(0)

    def get_smoothed_age(self) -> Optional[int]:
        """Return median age from history to filter outliers"""
        if not self.age_history:
            return None
        return int(np.median(self.age_history))

    def get_smoothed_gender(self) -> str:
        """Return most frequent gender from history, prioritized by registration if available"""
        # PRIORITY 1: Explicitly registered gender (if person is recognized)
        if self.name != "Unknown" and self.registered_gender != "Unknown":
            return self.registered_gender
            
        # PRIORITY 2: Majority vote from history
        if not self.gender_history:
            return "Unknown"
        
        from collections import Counter
        counts = Counter(self.gender_history)
        return counts.most_common(1)[0][0]

    def get_smoothed_emotion(self) -> str:
        """Return most frequent emotion from history"""
        if not self.emotion_history:
            return "Unknown"
        from collections import Counter
        counts = Counter(self.emotion_history)
        return counts.most_common(1)[0][0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "face_id": self.id,
            "name": self.name,
            "confidence": round(self.recognition_confidence, 2),
            "gender": self.get_smoothed_gender(),
            "age": self.get_smoothed_age(),
            "emotion": self.get_smoothed_emotion(),
            "anti_spoof": {
                "status": self.anti_spoof_status,
                "confidence": round(self.anti_spoof_confidence, 2),
                "is_live": self.is_live
            },
            "bbox": list(self.bbox),
            "duration": round(self.duration_seconds, 1),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "recognition_locked": self.recognition_locked
        }


def _compute_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two (x, y, w, h) bounding boxes"""
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    # Convert to (x1, y1, x2, y2)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    # Intersection
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


class MultiFaceTracker:
    """
    Track multiple faces using IoU + centroid hybrid matching.
    Uses Hungarian (linear_sum_assignment) for optimal assignment.
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.2):
        self.next_id = 1
        self.tracked_faces: OrderedDict[int, FaceTrack] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.start_time = time.time()

    def update(self, face_bboxes: List[Tuple[int, int, int, int]]) -> Dict[int, FaceTrack]:
        """
        Update tracker with new face detections.
        Returns Dict of {face_id: FaceTrack} for active faces.
        """
        current_time = time.time() - self.start_time

        # No detections → mark all as disappeared
        if len(face_bboxes) == 0:
            for face_id in list(self.tracked_faces.keys()):
                self.tracked_faces[face_id].mark_disappeared()
                if self.tracked_faces[face_id].disappeared_count > self.max_disappeared:
                    del self.tracked_faces[face_id]
            return self.tracked_faces

        # No existing tracks → register all as new
        if len(self.tracked_faces) == 0:
            for bbox in face_bboxes:
                self._register(bbox, current_time)
            return self.tracked_faces

        # Match existing tracks with new detections
        self._match_and_update(face_bboxes, current_time)
        return self.tracked_faces

    def _register(self, bbox: Tuple[int, int, int, int], timestamp: float) -> int:
        """Register a new face track"""
        face_id = self.next_id
        self.tracked_faces[face_id] = FaceTrack(face_id, bbox, timestamp)
        self.next_id += 1
        return face_id

    def _match_and_update(self, new_bboxes: List[Tuple[int, int, int, int]], timestamp: float):
        """Match new detections with existing tracked faces using IoU + centroid hybrid"""
        face_ids = list(self.tracked_faces.keys())
        num_tracks = len(face_ids)
        num_detections = len(new_bboxes)

        # Build cost matrix: rows=existing tracks, cols=new detections
        # Cost = 1 - combined_score  (lower cost = better match)
        cost_matrix = np.ones((num_tracks, num_detections), dtype=np.float64)

        for i, fid in enumerate(face_ids):
            track = self.tracked_faces[fid]
            for j, new_bbox in enumerate(new_bboxes):
                # IoU score
                iou = _compute_iou(track.bbox, new_bbox)

                # Centroid distance score (normalized by frame diagonal ~1000px)
                new_centroid = (new_bbox[0] + new_bbox[1] / 2.0, new_bbox[1] + new_bbox[3] / 2.0)
                # Use predicted centroid if we have velocity info
                dt = max(timestamp - (time.time() - self.start_time - track.last_seen), 0.033)
                pred = track.predicted_centroid(dt)
                dist = np.sqrt((pred[0] - new_centroid[0]) ** 2 + (pred[1] - new_centroid[1]) ** 2)
                centroid_score = max(0, 1.0 - dist / 500.0)  # Normalize: 500px = score 0

                # Size similarity score
                tw, th = track.bbox[2], track.bbox[3]
                nw, nh = new_bbox[2], new_bbox[3]
                size_ratio = min(tw * th, nw * nh) / max(tw * th, nw * nh, 1)

                # Combined score: IoU-heavy when overlapping, centroid when not
                if iou > 0.05:
                    combined = 0.6 * iou + 0.2 * centroid_score + 0.2 * size_ratio
                else:
                    combined = 0.1 * iou + 0.6 * centroid_score + 0.3 * size_ratio

                cost_matrix[i, j] = 1.0 - combined

        # Use Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            # Fallback: greedy matching if scipy not available
            row_indices, col_indices = self._greedy_assignment(cost_matrix)

        matched_tracks = set()
        matched_detections = set()

        for row, col in zip(row_indices, col_indices):
            # Only accept match if cost is reasonable (combined_score > 0.25)
            if cost_matrix[row, col] < 0.75:
                face_id = face_ids[row]
                self.tracked_faces[face_id].update_position(new_bboxes[col], timestamp)
                matched_tracks.add(row)
                matched_detections.add(col)

        # Mark unmatched tracks as disappeared
        for i in range(num_tracks):
            if i not in matched_tracks:
                face_id = face_ids[i]
                self.tracked_faces[face_id].mark_disappeared()
                if self.tracked_faces[face_id].disappeared_count > self.max_disappeared:
                    del self.tracked_faces[face_id]

        # Register unmatched detections as new tracks
        for j in range(num_detections):
            if j not in matched_detections:
                self._register(new_bboxes[j], timestamp)

    def _greedy_assignment(self, cost_matrix: np.ndarray):
        """Fallback greedy matching when scipy is not available"""
        rows, cols = [], []
        used_rows, used_cols = set(), set()
        flat = cost_matrix.flatten()
        sorted_indices = np.argsort(flat)

        for idx in sorted_indices:
            r = idx // cost_matrix.shape[1]
            c = idx % cost_matrix.shape[1]
            if r not in used_rows and c not in used_cols:
                rows.append(r)
                cols.append(c)
                used_rows.add(r)
                used_cols.add(c)
            if len(rows) >= min(cost_matrix.shape):
                break

        return np.array(rows), np.array(cols)

    def get_face(self, face_id: int) -> Optional[FaceTrack]:
        """Get a specific tracked face"""
        return self.tracked_faces.get(face_id)

    def get_all_faces(self) -> Dict[int, FaceTrack]:
        """Get all currently tracked faces"""
        return self.tracked_faces

    def get_face_count(self) -> int:
        """Get number of currently tracked faces"""
        return len(self.tracked_faces)

    def get_tracks_needing_recognition(self) -> List[FaceTrack]:
        """Get tracks that need recognition (new or low confidence)"""
        return [
            t for t in self.tracked_faces.values()
            if t.needs_recognition and t.disappeared_count == 0
        ]

    def reset(self):
        """Reset tracker (clear all faces)"""
        self.tracked_faces.clear()
        self.next_id = 1


class FaceLogger:
    """Log entry/exit events for tracked faces"""

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.active_faces: set = set()

    def log_entry(self, face_track: FaceTrack):
        """Log face entry event"""
        if face_track.id not in self.active_faces:
            self.logs.append({
                "event": "ENTRY",
                "face_id": face_track.id,
                "name": face_track.name,
                "timestamp": face_track.first_seen,
                "is_live": face_track.is_live
            })
            self.active_faces.add(face_track.id)

    def log_exit(self, face_track: FaceTrack):
        """Log face exit event"""
        if face_track.id in self.active_faces:
            self.logs.append({
                "event": "EXIT",
                "face_id": face_track.id,
                "name": face_track.name,
                "timestamp": face_track.last_seen,
                "duration_seconds": face_track.duration_seconds
            })
            self.active_faces.remove(face_track.id)

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent logs"""
        return self.logs[-limit:]

    def get_current_occupancy(self) -> int:
        """Get number of people currently present"""
        return len(self.active_faces)

    def clear_logs(self):
        """Clear all logs"""
        self.logs.clear()
        self.active_faces.clear()
