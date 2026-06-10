"""
Frame Processor - Core AI Engine
Handles all facial intelligence features with toggle support
"""
import cv2
import numpy as np
import mediapipe as mp
import threading
from typing import Dict, Any, Tuple, List, Optional
import time
# from deepface import DeepFace
# from sklearn.metrics.pairwise import cosine_similarity
from face_recognition_engine import FaceRecognitionEngine
from ar_filters import ARFilterEngine  # NEW: AR Filter Engine
from anti_spoofing import AntiSpoofingEngine  # NEW: Anti-spoofing security
from face_tracker import MultiFaceTracker, FaceLogger  # NEW: Multi-face tracking

class FrameProcessor:
    def __init__(self):
        # Core feature flags
        self.features = {
            "face_detection": True,
            "face_landmarks": True,
            "face_recognition": True,
            "multi_face_tracking": True,      # CRITICAL: Force-enable for labels
            "emotion": True,
            "age_gender": True,
            "head_pose": True,
            "eye_gaze": True,
            "blink_detection": True,
            "attention": True,
            "fatigue": True,
            "stress": True,
            "anti_spoofing": True,
            "background_removal": False,
            "style_filter": False
        }
        
        # Feature Execution Order (sequential pipeline)
        self.feature_execution_order = [
            'face_detection',      # Step 1: Mandatory base
            'face_landmarks',      # Step 2: Unlock pose/gaze
            'anti_spoofing',       # Step 3: Anti-spoofing BEFORE recognition (security)
            'face_recognition',    # Step 4: Name labeling (after anti-spoof check)
            'age_gender',          # Step 5: Demographics
            'emotion',             # Step 6: Emotion
            'head_pose',           # Step 7: Pose estimation
            'eye_gaze',            # Step 8: Gaze tracking
            'blink_detection',     # Step 9: Blink
            'attention',           # Step 10: Attention
            'fatigue',             # Step 11: Fatigue
            'stress',              # Step 12: Stress
            'background_removal',  # Step 13: Effects
        ]
        
        # Feature Dependencies (features that require other features)
        self.feature_dependencies = {
            'face_landmarks': ['face_detection'],
            'face_recognition': ['face_detection'],
            'age_gender': ['face_detection'],
            'emotion': ['face_detection'],
            'head_pose': ['face_landmarks'],
            'eye_gaze': ['face_landmarks'],
            'blink_detection': ['face_landmarks'],
            'attention': ['eye_gaze'],
            'stress': ['emotion'],
            'fatigue': ['blink_detection'],
            'anti_spoofing': ['face_landmarks', 'blink_detection'],
            'multi_face_tracking': ['face_detection'],
            'background_removal': ['face_detection'],
        }
        
        # Threading & Synchronization
        self.recognition_lock = threading.Lock()
        self.deepface_lock = threading.Lock() # Global lock for DeepFace operations to prevent TF session conflicts
        self.is_processing_recognition = False
        self.is_processing_deepface = False
        self.last_recognition_update = 0
        self.last_deepface_update = 0
        
        # Initialize MediaPipe (Guarded)
        if mp:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_face_detection = mp.solutions.face_detection
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=5,
                    refine_landmarks=True,  # Required for iris landmarks (gaze tracking)
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
                
                # MediaPipe Selfie Segmentation for background removal
                self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
                self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=1
                )
                
                # Drawing utilities
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                print("[OK] MediaPipe initialized")
            except Exception as e:
                print(f"[WARN] MediaPipe initialization error: {e}")
                self.face_mesh = None
                self.face_detection = None
                self.selfie_segmentation = None
        else:
            print("[ERROR] MediaPipe not available. Face detection/mesh disabled.")
            self.face_mesh = None
            self.face_detection = None
            self.selfie_segmentation = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
            self.features["face_detection"] = False
            self.features["background_removal"] = False
        # Blink detection state
        self.blink_counter = 0
        self.eye_ar_threshold = 0.21
        self.prev_ear = 0.3
        self.was_blinking = False  # Proper instance variable for blink state tracking
        
        # Liveness detection state
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # DeepFace cache (to avoid running on every frame)
        self.deepface_results = {}  # Key: face_id (tracked via index for simplicity), Value: result
        self.last_deepface_update = 0
        
        # Face Recognition Engine
        self.face_recognition_engine = None
        self.recognition_results = {}  # Cache for recognition results
        self.last_recognition_update = 0
        self.recognition_update_interval = 1.0  # Increase to 1.0 seconds for performance
        self.is_processing_recognition = False
        
        # Try to initialize face recognition
        try:
            import os as _os
            _backend_dir = _os.path.dirname(_os.path.abspath(__file__))
            _dataset_dir = _os.path.join(_backend_dir, 'dataset')
            self.face_recognition_engine = FaceRecognitionEngine(
                model='Facenet512',
                threshold=0.75,
                dataset_root=_dataset_dir
            )
            print("[OK] Face Recognition Engine initialized")
        except Exception as e:
            print(f"[WARN] Face Recognition Engine initialization failed: {e}")
            self.features["face_recognition"] = False
        self.deepface_update_interval = 2.0  # Increase to 2.0 seconds for performance
        
        # AR Filter Engine (NEW)
        try:
            self.ar_filter_engine = ARFilterEngine()
            print("[OK] AR Filter Engine initialized")
        except Exception as e:
            print(f"[WARN] AR Filter Engine initialization failed: {e}")
            self.ar_filter_engine = None
        
        # NEW: Anti-Spoofing Engine
        try:
            self.anti_spoofing_engine = AntiSpoofingEngine()
            print("[OK] Anti-Spoofing Engine initialized")
        except Exception as e:
            print(f"[WARN] Anti-Spoofing Engine initialization failed: {e}")
            self.anti_spoofing_engine = None
            self.features["anti_spoofing"] = False
        
        # NEW: Multi-Face Tracker
        try:
            self.face_tracker = MultiFaceTracker(max_disappeared=30)
            self.face_logger = FaceLogger()
            print("[OK] Multi-Face Tracker initialized")
        except Exception as e:
            print(f"[WARN] Multi-Face Tracker initialization failed: {e}")
            self.face_tracker = None
            self.face_logger = None
            self.features["multi_face_tracking"] = False
        
    def toggle_feature(self, feature: str, enabled: bool):
        """Toggle a specific feature on/off with dependency validation"""
        if feature not in self.features:
            return False
            
        # If enabling, check dependencies
        if enabled:
            deps = self.feature_dependencies.get(feature, [])
            missing_deps = [dep for dep in deps if not self.features.get(dep, False)]
            
            if missing_deps:
                print(f"[ERROR] Cannot enable '{feature}': Missing dependencies {missing_deps}")
                return False
        
        # If disabling, auto-disable dependent features
        if not enabled:
            for feat, deps in self.feature_dependencies.items():
                if feature in deps and self.features.get(feat, False):
                    print(f"[WARN] Auto-disabling '{feat}' due to '{feature}' being disabled")
                    self.features[feat] = False
        
        self.features[feature] = enabled
        print(f"[OK] Feature '{feature}' {'enabled' if enabled else 'disabled'}")

        # When face recognition is toggled ON → re-queue all active tracks
        if feature == "face_recognition" and enabled and self.face_tracker:
            for track in self.face_tracker.tracked_faces.values():
                track.needs_recognition = True
                track.recognition_locked = False
                track.recognition_attempts = 0
                track.last_recognition_attempt = 0.0
            print("Face recognition enabled - all active tracks re-queued for recognition")

        return True
            
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features"""
        return [k for k, v in self.features.items() if v]
    
    def get_all_features(self) -> Dict[str, bool]:
        """Get all features and their status"""
        return self.features.copy()
    
    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame with all enabled AI features
        Returns: (processed_frame, attributes_dict)
        """
        if frame is None:
            return frame, {}
        
        processed_frame = frame.copy()
        colormode_frame = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        attributes = {
            "faces": [],
            "fps": 0,
            "timestamp": time.time()
        }
        
        # Background Removal
        if self.features["background_removal"]:
            processed_frame = self._apply_background_removal(processed_frame, rgb_frame)
            colormode_frame = processed_frame # Update for subsequent drawing
        
        # ===================================================================
        # STEP 1: FACE DETECTION → get raw bounding boxes
        # ===================================================================
        detection_bboxes = []   # List of (x, y, w, h)
        detection_crops = {}    # idx → face_crop (BGR)
        
        if self.features["face_detection"] and self.face_detection:
            detection_results = self.face_detection.process(rgb_frame)
            
            if detection_results.detections:
                for det_idx, detection in enumerate(detection_results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(w - x, int(bbox.width * w))
                    height = min(h - y, int(bbox.height * h))
                    
                    detection_bboxes.append((x, y, width, height))
                    
                    # Store crop for recognition/deepface (indexed by detection order)
                    face_crop = frame[y:y+height, x:x+width]
                    if face_crop.size > 0:
                        detection_crops[det_idx] = face_crop

        # ===================================================================
        # STEP 2: MULTI-FACE TRACKING → assign stable IDs to detections
        # ===================================================================
        tracked_faces = {}
        track_to_bbox = {}  # track_id → (det_idx, bbox)
        
        if self.face_tracker:
            tracked_faces = self.face_tracker.update(detection_bboxes)
            
            # EXCLUSIVE assignment: each detection can only be assigned to ONE track
            assigned_det_indices = set()
            iou_candidates = []
            for track_id, track in tracked_faces.items():
                if track.disappeared_count > 0:
                    continue
                for det_idx, det_bbox in enumerate(detection_bboxes):
                    iou = self._bbox_overlap(track.bbox, det_bbox)
                    if iou > 0.3: # Increased threshold for stricter matching
                        iou_candidates.append((iou, track_id, det_idx))
            
            iou_candidates.sort(key=lambda t: t[0], reverse=True)
            for iou, track_id, det_idx in iou_candidates:
                if track_id not in track_to_bbox and det_idx not in assigned_det_indices:
                    track_to_bbox[track_id] = (det_idx, detection_bboxes[det_idx])
                    assigned_det_indices.add(det_idx)

        # ===================================================================
        # STEP 2.5: DETECTION-ONLY FALLBACK (If tracking is off or fails)
        # ===================================================================
        # If tracking is disabled or a detection wasn't matched, we still draw a box
        if not self.features["multi_face_tracking"] or len(assigned_det_indices) < len(detection_bboxes):
            for det_idx, det_bbox in enumerate(detection_bboxes):
                if det_idx not in assigned_det_indices:
                    # Draw a simple box for untracked detections
                    x, y, w, h = det_bbox
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
                    cv2.putText(processed_frame, "Detecting...", (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ===================================================================
        # STEP 3: FACE RECOGNITION — only for NEW / unrecognized tracks
        # ===================================================================
        if self.features["face_recognition"] and self.face_recognition_engine:
            for track_id, track in tracked_faces.items():
                if track.disappeared_count > 0:
                    continue
                
                # Only run recognition if this track needs it
                if not track.needs_recognition:
                    continue
                
                # Cooldown: don't spam recognition for the same track
                now = time.time()
                if now - track.last_recognition_attempt < 1.5:
                    continue
                
                # Get the face crop for this track (exclusively assigned)
                if track_id not in track_to_bbox:
                    continue
                det_idx, _ = track_to_bbox[track_id]
                if det_idx not in detection_crops:
                    continue
                
                face_crop = detection_crops[det_idx]
                
                # Run recognition asynchronously (only one at a time)
                if not self.is_processing_recognition:
                    self.is_processing_recognition = True
                    track.last_recognition_attempt = now
                    # Mark that this track is pending — avoids double-triggering
                    track.needs_recognition = False  # Will be re-set to True on error
                    threading.Thread(
                        target=self._run_recognition_task,
                        args=(face_crop.copy(), track_id)
                    ).start()
        else:
            # If recognition is OFF, mark all tracks as "done" so they display correctly
            for track in tracked_faces.values():
                if track.needs_recognition and track.name == "Unknown":
                    track.needs_recognition = False

        # ===================================================================
        # STEP 4: FACE MESH + LANDMARKS (Required for AR filters)
        # ===================================================================
        mesh_results = None
        if self.face_mesh:
            mesh_results = self.face_mesh.process(rgb_frame)
        
        if mesh_results and mesh_results.multi_face_landmarks:
            h_frame, w_frame = frame.shape[:2]
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Calculate mesh bbox to match with tracked face
                x_coords = [lm.x * w_frame for lm in face_landmarks.landmark]
                y_coords = [lm.y * h_frame for lm in face_landmarks.landmark]
                mesh_bbox = (int(min(x_coords)), int(min(y_coords)), 
                             int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords)))
                
                # Match with tracked faces
                matched_tid = None
                best_iou = 0.3
                for tid, t in tracked_faces.items():
                    if t.disappeared_count == 0:
                        iou = self._bbox_overlap(mesh_bbox, tuple(t.bbox))
                        if iou > best_iou:
                            best_iou = iou
                            matched_tid = tid
                
                if matched_tid:
                    track = tracked_faces[matched_tid]
                    # Update all landmark-based attributes
                    if self.features["head_pose"]:
                        track.head_pose = self._estimate_head_pose(face_landmarks, frame.shape)
                    if self.features["eye_gaze"]:
                        track.gaze = self._estimate_gaze(face_landmarks, frame.shape)
                    if self.features["blink_detection"]:
                        left_eye  = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                        right_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                        track.blink = self._detect_blink(left_eye, right_eye, track=track)
                    if self.features["attention"]:
                        track.attention = self._estimate_attention(track.head_pose or {}, track.gaze or {})
                    if self.features["fatigue"]:
                        track.fatigue = self._detect_fatigue(track.blink or {})
                    if self.features["stress"]:
                        track.stress = self._estimate_stress(track.get_smoothed_emotion(), 
                                                            (track.fatigue or {}).get("score", 0))
                    if self.features["anti_spoofing"] and self.anti_spoofing_engine:
                        track.anti_spoof = self.anti_spoofing_engine.verify_liveness(frame, face_landmarks, track.blink or {})

        # ===================================================================
        # STEP 5: AR FILTERS (Before Drawing HUD)
        # ===================================================================
        if self.ar_filter_engine and self.ar_filter_engine.current_filter != "none":
            if mesh_results and mesh_results.multi_face_landmarks:
                processed_frame = self.ar_filter_engine.apply_filter(
                    processed_frame,
                    mesh_results.multi_face_landmarks,
                    self.ar_filter_engine.current_filter
                )

        # ===================================================================
        # STEP 6: BUILD faces_data + DRAW bounding boxes + names
        # ===================================================================
        faces_data = []
        
        for track_id, track in tracked_faces.items():
            try:
                if track.disappeared_count > 0:
                    continue  # Don't draw disappeared faces
                
                x, y, w_box, h_box = track.bbox
                x, y = max(0, x), max(0, y)
                
                face_data = {
                    "bbox": [x, y, w_box, h_box],
                    "confidence": track.recognition_confidence,
                    "id": track_id,
                    "face_id": track_id,
                    "name": track.name,
                    "recognition_confidence": track.recognition_confidence,
                    "registered_gender": track.registered_gender,
                    "age": track.get_smoothed_age(),
                    "gender": track.get_smoothed_gender(),
                    "emotion": track.get_smoothed_emotion(),
                    "duration": round(track.duration_seconds, 1),
                    "anti_spoof": track.anti_spoof  # Security status per face
                }
                
                # Draw bounding box — color driven by emotion (if available)
                _EMOTION_COLORS = {
                    "happy":    (0, 220, 80),    # Green
                    "neutral":  (180, 180, 180), # Light gray
                    "sad":      (200, 80, 0),    # Dark blue-ish
                    "angry":    (0, 0, 230),     # Red
                    "fear":     (180, 0, 220),   # Purple
                    "surprise": (0, 165, 255),   # Orange
                    "disgust":  (60, 120, 0),    # Dark green
                }
                emotion_str = track.get_smoothed_emotion()
                if emotion_str and emotion_str.lower() in _EMOTION_COLORS:
                    box_color = _EMOTION_COLORS[emotion_str.lower()]
                elif track.recognition_locked:
                    box_color = (0, 255, 0)   # Green — confirmed identity
                else:
                    box_color = (255, 200, 0) # Blue-white — tracking only
    
                # Draw box with slightly thicker line when emotion is known
                box_thickness = 3 if emotion_str else 2
                cv2.rectangle(processed_frame, (x, y), (x + w_box, y + h_box), box_color, box_thickness)
    
                # --- Robust Label Drawing (Matching Screenshot Layout) ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                attr_font_scale = 0.65
                attr_thickness = 2
                line_gap = 28  # vertical gap between attribute lines

                # --- Per-Track Async DeepFace Analysis ---
                if track_id in track_to_bbox:
                    det_idx, _ = track_to_bbox[track_id]
                    recognition_ready = not track.needs_recognition or not self.features["face_recognition"]
                    now_df = time.time()

                    run_deepface = (
                        (self.features["age_gender"] or self.features["emotion"]) and
                        not track.is_processing_deepface and
                        recognition_ready and
                        (now_df - track.last_deepface_update > self.deepface_update_interval)
                    )

                    if run_deepface and det_idx in detection_crops:
                        actions = []
                        if self.features["age_gender"]:
                            actions.extend(['age', 'gender'])
                        if self.features["emotion"]:
                            actions.append('emotion')
                        if actions:
                            track.is_processing_deepface = True
                            threading.Thread(
                                target=self._run_deepface_task,
                                args=(detection_crops[det_idx].copy(), track_id, actions)
                            ).start()

                res = track.deepface_result

                # -- ABOVE BOX: Build labels ---------------------------------
                above_y = y  # bottom edge of the text region above box
                demo_label = None
                if self.features.get("age_gender"):
                    gender_str = track.get_smoothed_gender()
                    age_val = track.get_smoothed_age()
                    if age_val:
                        demo_label = f"{gender_str}, Age: {age_val}"
                    else:
                        demo_label = f"{gender_str}"

                # -- Identity label: [ NAME CONF% ] format ------------------
                # PRIORITY: Only show names if Face Recognition feature is ON
                if self.features.get("face_recognition"):
                    if track.name != "Unknown":
                        id_label = f"[ {track.name}  {track.recognition_confidence:.0f}% ]"
                        id_bg_color = (0, 200, 0)    # Green for known person
                        id_text_color = (255, 255, 255)
                    else:
                        id_label = "[ Unknown ]"
                        id_bg_color = (60, 60, 60)   # Dark gray for unknown
                        id_text_color = (200, 200, 200)
                else:
                    # Recognition is OFF — show generic label
                    id_label = f"Face #{track_id}"
                    id_bg_color = (100, 100, 100) # Gray
                    id_text_color = (255, 255, 255)

                # Measure sizes
                (id_w, id_h), _ = cv2.getTextSize(id_label, font, font_scale, thickness)

                # --- Boundary Check for Top Labels ---
                demo_h = 15 if demo_label else 0
                required_top_space = id_h + 14 + demo_h + 8

                # If box is too close to top, move labels inside the box
                if y - required_top_space < 0:
                    above_y = y + required_top_space + 5
                else:
                    above_y = y

                # Draw identity label with colored background pill
                id_bg_top    = above_y - id_h - 12
                id_bg_bottom = above_y
                # Rounded-corner approximation using two overlapping rectangles
                cv2.rectangle(processed_frame,
                              (x, id_bg_top), (x + id_w + 14, id_bg_bottom),
                              id_bg_color, -1)
                cv2.putText(processed_frame, id_label,
                            (x + 7, id_bg_bottom - 5),
                            font, font_scale, id_text_color, thickness)

                # Draw demographics label ABOVE the identity label
                if demo_label:
                    demo_y = id_bg_top - 8
                    cv2.putText(processed_frame, demo_label,
                                (x, demo_y),
                                font, attr_font_scale, (0, 230, 255), attr_thickness)

                # Add all current metrics to face_data for frontend
                face_data.update({
                    "head_pose": track.head_pose,
                    "gaze": track.gaze,
                    "blink": track.blink,
                    "attention": track.attention,
                    "fatigue": track.fatigue,
                    "stress": track.stress
                })

                # Blink indicator (flash above box)
                if track.blink and track.blink.get("is_blinking"):
                    cv2.putText(processed_frame, "BLINK", (x, y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                faces_data.append(face_data)
            except Exception as e:
                print(f"[ERROR] Track drawing failure: {e}")
                continue

        # Draw Landmarks (Mesh) AFTER filtering
        if self.features.get("face_landmarks") and mesh_results and mesh_results.multi_face_landmarks:
            h_f, w_f = frame.shape[:2]
            for face_landmarks in mesh_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=processed_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Specialized Drawing: Head Pose & Gaze
                if self.features["head_pose"]:
                    self._draw_head_pose(processed_frame, face_landmarks, 
                                        self._estimate_head_pose(face_landmarks, (h_f, w_f)), (h_f, w_f))
                if self.features["eye_gaze"]:
                    self._draw_gaze(processed_frame, face_landmarks, 
                                   self._estimate_gaze(face_landmarks, (h_f, w_f)), (h_f, w_f))

        # ===================================================================
        # STEP 6: CLEANUP + FPS + ATTRIBUTES
        # ===================================================================
        # Clean up stale cache entries for tracks that no longer exist
        active_track_ids = set(tracked_faces.keys())
        for stale_id in list(self.deepface_results.keys()):
            if stale_id not in active_track_ids:
                del self.deepface_results[stale_id]

        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        attributes["faces"] = faces_data
        attributes["fps"] = round(self.fps, 1)
        attributes["face_count"] = len(faces_data)
        
        # Add multi-face tracking info
        if self.features["multi_face_tracking"] and tracked_faces:
            attributes["tracked_faces"] = {
                fid: {
                    "name": ft.name,
                    "duration": round(ft.duration_seconds, 1),
                    "is_live": ft.is_live
                }
                for fid, ft in tracked_faces.items()
                if ft.disappeared_count == 0
            }
        
        # Draw FPS
        cv2.putText(processed_frame, f"FPS: {attributes['fps']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face count
        if self.features["multi_face_tracking"]:
            cv2.putText(processed_frame, f"Faces: {attributes['face_count']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        return processed_frame, attributes






    def _run_deepface_task(self, face_img, track_id, actions):
        """Background worker for DeepFace Analysis — stores result on FaceTrack object"""
        try:
            from deepface import DeepFace
        except ImportError:
            print("[ERROR] DeepFace not installed. Skipping analysis.")
            return

        track = self.face_tracker.get_face(track_id) if self.face_tracker else None
        try:
            with self.deepface_lock:
                print(f"DeepFace: Starting analysis for track {track_id} actions={actions}")
                results = DeepFace.analyze(
                    img_path=face_img,
                    actions=actions,
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
            if results and track:
                res = results[0]
                track.deepface_result = res
                track.last_deepface_update = time.time()
                
                # Update demographic history for smoothing
                track.update_demographics(
                    age=res.get("age"),
                    gender=res.get("dominant_gender"),
                    emotion=res.get("dominant_emotion")
                )
                
                print(f"DeepFace: Track {track_id} → age={res.get('age')} (smoothed={track.get_smoothed_age()}), "
                      f"gender={res.get('dominant_gender')} (smoothed={track.get_smoothed_gender()}), "
                      f"emotion={res.get('dominant_emotion')}")
            elif not track:
                print(f"DeepFace: Track {track_id} gone, discarding result")
        except Exception as e:
            print(f"DeepFace error for track {track_id}: {e}")
        finally:
            if track:
                track.is_processing_deepface = False
    
    def _run_recognition_task(self, face_img, track_id):
        """Background worker for Face Recognition — stores result on the FaceTrack object"""
        try:
            print(f"FaceRec: Starting recognition for track ID {track_id}")
            # Recognition inside lock because it uses DeepFace.represent
            with self.deepface_lock:
                 name, confidence, gender = self.face_recognition_engine.recognize_face(face_img, skip_detection=False)
            
            # Store result directly on the FaceTrack object (not in a shared dict)
            if self.face_tracker:
                track = self.face_tracker.get_face(track_id)
                if track:
                    track.set_recognition(name, confidence, gender)
                    print(f"FaceRec: Track ID {track_id} → {name} ({confidence:.1f}%) Gender: {gender} | locked={track.recognition_locked}")
                else:
                    print(f"FaceRec: Track ID {track_id} no longer exists, discarding result")
            
            self.last_recognition_update = time.time()
        except Exception as e:
            print(f"FaceRec error for track {track_id}: {e}")
            # On error, mark the track so it can retry
            if self.face_tracker:
                track = self.face_tracker.get_face(track_id)
                if track:
                    track.needs_recognition = True  # Allow retry
        finally:
            self.is_processing_recognition = False



    
    def _apply_background_removal(self, frame: np.ndarray, rgb_frame: np.ndarray) -> np.ndarray:
        """Remove background using MediaPipe Selfie Segmentation"""
        try:
            if not self.selfie_segmentation:
                return frame
            results = self.selfie_segmentation.process(rgb_frame)
            
            if results.segmentation_mask is not None:
                # Create a binary mask
                mask = results.segmentation_mask > 0.5
                mask = mask.astype(np.uint8) * 255
                
                # Create a blurred background
                blurred_bg = cv2.GaussianBlur(frame, (55, 55), 0)
                
                # Combine foreground and blurred background
                mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                foreground = cv2.bitwise_and(frame, mask_3channel)
                background = cv2.bitwise_and(blurred_bg, cv2.bitwise_not(mask_3channel))
                
                return cv2.add(foreground, background)
        except Exception as e:
            print(f"Background removal error: {e}")
        
        return frame
    
    def _estimate_head_pose(self, landmarks, image_shape) -> Dict[str, float]:
        """Estimate head pose (yaw, pitch, roll)"""
        h, w, _ = image_shape
        
        # Key points for head pose
        nose_tip = landmarks.landmark[1]
        chin = landmarks.landmark[152]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        left_mouth = landmarks.landmark[61]
        right_mouth = landmarks.landmark[291]
        
        # Convert to pixel coordinates
        points_2d = np.array([
            [nose_tip.x * w, nose_tip.y * h],
            [chin.x * w, chin.y * h],
            [left_eye.x * w, left_eye.y * h],
            [right_eye.x * w, right_eye.y * h],
            [left_mouth.x * w, left_mouth.y * h],
            [right_mouth.x * w, right_mouth.y * h]
        ], dtype=np.float64)
        
        # 3D model points
        points_3d = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye
            [225.0, 170.0, -135.0],    # Right eye
            [-150.0, -150.0, -125.0],  # Left mouth
            [150.0, -150.0, -125.0]    # Right mouth
        ], dtype=np.float64)
        
        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Calculate Euler angles
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]
        
        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll)
        }
    
    def _draw_head_pose(self, frame: np.ndarray, landmarks, head_pose: Dict, image_shape):
        """Draw head pose visualization"""
        h, w, _ = image_shape
        nose_tip = landmarks.landmark[1]
        
        x = int(nose_tip.x * w)
        y = int(nose_tip.y * h)
        
        # Draw text
        text = f"Y:{head_pose['yaw']:.0f} P:{head_pose['pitch']:.0f} R:{head_pose['roll']:.0f}"
        cv2.putText(frame, text, (x - 50, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _estimate_gaze(self, landmarks, image_shape) -> Dict[str, str]:
        """Estimate eye gaze direction using iris landmarks (requires refine_landmarks=True)"""
        try:
            # Iris landmarks — only available when refine_landmarks=True
            left_iris  = landmarks.landmark[468]  # Left iris center
            right_iris = landmarks.landmark[473]  # Right iris center

            # Eye corner landmarks
            left_eye_left   = landmarks.landmark[33]
            left_eye_right  = landmarks.landmark[133]
            left_eye_top    = landmarks.landmark[159]
            left_eye_bottom = landmarks.landmark[145]

            right_eye_left   = landmarks.landmark[362]
            right_eye_right  = landmarks.landmark[263]
            right_eye_top    = landmarks.landmark[386]
            right_eye_bottom = landmarks.landmark[374]

            # Horizontal ratio (left-right gaze)
            left_h_width  = left_eye_right.x  - left_eye_left.x
            right_h_width = right_eye_right.x - right_eye_left.x

            left_h_ratio  = (left_iris.x  - left_eye_left.x)  / (left_h_width  + 1e-6)
            right_h_ratio = (right_iris.x - right_eye_left.x) / (right_h_width + 1e-6)
            avg_h_ratio   = (left_h_ratio + right_h_ratio) / 2.0

            # Vertical ratio (up-down gaze)
            left_v_height  = left_eye_bottom.y  - left_eye_top.y
            right_v_height = right_eye_bottom.y - right_eye_top.y

            left_v_ratio  = (left_iris.y  - left_eye_top.y)  / (left_v_height  + 1e-6)
            right_v_ratio = (right_iris.y - right_eye_top.y) / (right_v_height + 1e-6)
            avg_v_ratio   = (left_v_ratio + right_v_ratio) / 2.0

            # Determine horizontal direction
            if avg_h_ratio < 0.38:
                h_dir = "LEFT"
            elif avg_h_ratio > 0.62:
                h_dir = "RIGHT"
            else:
                h_dir = "CENTER"

            # Determine vertical direction
            if avg_v_ratio < 0.35:
                v_dir = "UP"
            elif avg_v_ratio > 0.65:
                v_dir = "DOWN"
            else:
                v_dir = "CENTER"

            # Combine into a single direction label
            if h_dir == "CENTER" and v_dir == "CENTER":
                direction = "CENTER"
            elif h_dir == "CENTER":
                direction = v_dir
            elif v_dir == "CENTER":
                direction = h_dir
            else:
                direction = f"{v_dir}-{h_dir}"

            return {
                "direction": direction,
                "h_ratio": round(float(avg_h_ratio), 3),
                "v_ratio": round(float(avg_v_ratio), 3),
            }
        except (IndexError, ZeroDivisionError) as e:
            # Fallback if iris landmarks unavailable
            return {"direction": "CENTER", "h_ratio": 0.5, "v_ratio": 0.5}
    
    def _draw_gaze(self, frame: np.ndarray, landmarks, gaze: Dict, image_shape):
        """Draw gaze direction"""
        h, w, _ = image_shape
        nose = landmarks.landmark[1]
        
        x = int(nose.x * w)
        y = int(nose.y * h)
        
        cv2.putText(frame, f"Gaze: {gaze['direction']}", (x + 20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _detect_blink(self, left_eye_landmarks, right_eye_landmarks, track=None) -> Dict[str, Any]:
        """
        Detect eye blinks using Eye Aspect Ratio (EAR).
        Uses per-track state if `track` is provided, else falls back to self.* (single-face mode).
        Returns dict with left_ear, right_ear, avg_ear, is_blinking, count.
        """
        def eye_aspect_ratio(eye_lms):
            v1 = np.linalg.norm(
                np.array([eye_lms[1].x, eye_lms[1].y]) -
                np.array([eye_lms[5].x, eye_lms[5].y])
            )
            v2 = np.linalg.norm(
                np.array([eye_lms[2].x, eye_lms[2].y]) -
                np.array([eye_lms[4].x, eye_lms[4].y])
            )
            h = np.linalg.norm(
                np.array([eye_lms[0].x, eye_lms[0].y]) -
                np.array([eye_lms[3].x, eye_lms[3].y])
            )
            if h < 1e-6:
                return 0.3
            return (v1 + v2) / (2.0 * h)

        left_ear  = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        avg_ear   = (left_ear + right_ear) / 2.0

        EAR_THRESHOLD = 0.21
        is_blinking = avg_ear < EAR_THRESHOLD

        # Use per-track state if available, else fall back to global self.*
        if track is not None:
            if is_blinking and not track.was_blinking:
                track.blink_counter += 1
            track.was_blinking = is_blinking
            blink_count = track.blink_counter
        else:
            if is_blinking and not self.was_blinking:
                self.blink_counter += 1
            self.was_blinking = is_blinking
            blink_count = self.blink_counter

        return {
            "left_ear":   round(float(left_ear),  3),
            "right_ear":  round(float(right_ear), 3),
            "avg_ear":    round(float(avg_ear),   3),
            "is_blinking": bool(is_blinking),
            "count":      blink_count,
        }
    
    def _estimate_attention(self, head_pose: Dict, gaze: Dict) -> Dict[str, Any]:
        """Estimate attention level based on head pose and gaze"""
        if not head_pose or not gaze:
            return {"level": "unknown", "score": 0}
        
        # Check if looking at screen (centered gaze and forward head pose)
        yaw = abs(head_pose.get("yaw", 0))
        pitch = abs(head_pose.get("pitch", 0))
        gaze_dir = gaze.get("direction", "CENTER")
        
        score = 100
        
        # Penalize for head rotation
        if yaw > 20:
            score -= min(yaw, 50)
        if pitch > 15:
            score -= min(pitch, 30)
        
        # Penalize for gaze direction
        if gaze_dir != "CENTER":
            score -= 20
        
        score = max(0, min(100, score))
        
        if score > 70:
            level = "HIGH"
        elif score > 40:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return {"level": level, "score": float(score)}
    
    def _detect_fatigue(self, blink_data: Dict) -> Dict[str, Any]:
        """Detect fatigue based on blink rate and eye closure"""
        if not blink_data:
            return {"level": "unknown", "score": 0}

        # Use avg_ear (correct key from _detect_blink output)
        ear = blink_data.get("avg_ear", blink_data.get("ear", 0.3))
        blink_count = blink_data.get("count", 0)

        fatigue_score = 0

        # Low EAR indicates drowsiness / heavy eyelids
        if ear < 0.18:
            fatigue_score += 50
        elif ear < 0.22:
            fatigue_score += 30
        elif ear < 0.26:
            fatigue_score += 10

        # High blink count can indicate fatigue
        if blink_count > 40:
            fatigue_score += 30
        elif blink_count > 20:
            fatigue_score += 15

        fatigue_score = min(100, fatigue_score)

        if fatigue_score > 60:
            level = "HIGH"
        elif fatigue_score > 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {"level": level, "score": float(fatigue_score)}
    
    def _estimate_stress(self, emotion: str, fatigue_score: float) -> Dict[str, Any]:
        """Estimate stress based on emotion and fatigue markers (Simplified Heuristic)"""
        stress_score = 0
        
        # Emotion contributes to stress
        high_stress_emotions = ["angry", "fear", "sad", "disgust"]
        if emotion in high_stress_emotions:
            stress_score += 60
        elif emotion == "neutral":
            stress_score += 10
        elif emotion == "surprise":
            stress_score += 30
            
        # Fatigue contributes to stress
        stress_score += (fatigue_score * 0.4)
        
        stress_score = min(100, stress_score)
        
        if stress_score > 70:
            level = "HIGH"
        elif stress_score > 40:
            level = "MEDIUM"
        else:
            level = "LOW"
            
        return {"level": level, "score": float(stress_score)}
    
    
    
    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Process an entire video file"""
        results = []
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            _, attributes = self.process_frame(frame)
            attributes["frame_index"] = frame_idx
            results.append(attributes)
            
            frame_idx += 1
        
        cap.release()
        return results

