"""
Anti-Spoofing Engine - Advanced AI Security
Detects real vs fake faces using multiple security checks
"""
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple
import time


class BlinkDetector:
    """Enhanced blink detection with frequency and pattern analysis"""
    
    def __init__(self):
        self.blink_history = []
        self.max_history = 30  # Last 30 blinks
        self.last_blink_time = 0
        self.consecutive_frames_closed = 0
        
    def analyze_blink(self, left_ear: float, right_ear: float, timestamp: float) -> Dict[str, Any]:
        """
        Analyze blink patterns for naturalness
        
        Returns:
            Dict with 'passed', 'score', 'reason'
        """
        EAR_THRESHOLD = 0.21
        
        # Both eyes should blink together (synchronous)
        both_closed = left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD
        
        if both_closed:
            self.consecutive_frames_closed += 1
        else:
            # Blink ended
            if self.consecutive_frames_closed >= 2:  # Minimum 2 frames for valid blink
                blink_duration = self.consecutive_frames_closed / 30.0  # Assuming 30 FPS
                
                # Natural blink: 100-400ms (3-12 frames at 30fps)
                if 0.067 <= blink_duration <= 0.4:  # 2-12 frames
                    self.blink_history.append({
                        'timestamp': timestamp,
                        'duration': blink_duration
                    })
                    
                    # Keep only recent blinks
                    self.blink_history = self.blink_history[-self.max_history:]
            
            self.consecutive_frames_closed = 0
        
        # Analyze blink frequency
        recent_blinks = [b for b in self.blink_history if timestamp - b['timestamp'] < 60]  # Last minute
        
        if len(recent_blinks) == 0:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "No blinks detected"
            }
        
        # Natural blink rate: 15-20 per minute
        blinks_per_minute = len(recent_blinks) * (60.0 / min(timestamp - self.blink_history[0]['timestamp'] if self.blink_history else 1, 60))
        
        # Check if blink rate is natural
        is_natural_rate = 10 <= blinks_per_minute <= 30
        
        # Calculate score based on blink rate
        if is_natural_rate:
            score = min(len(recent_blinks) / 5, 1.0)  # Reach 1.0 after 5 blinks
        else:
            score = 0.3
        
        return {
            "passed": len(recent_blinks) >= 2 and is_natural_rate,
            "score": score,
            "reason": f"Blinks: {len(recent_blinks)}" if is_natural_rate else "Abnormal blink rate"
        }


class TextureAnalyzer:
    """Analyzes skin texture to detect printed photos"""
    
    def analyze_face_texture(self, face_region: np.ndarray) -> Dict[str, Any]:
        """
        Analyze skin texture quality
        Real skin has depth, pores, micro-textures
        Photos are flat and uniform
        """
        if face_region is None or face_region.size == 0:
            return {"passed": False, "score": 0.0, "reason": "Invalid face region"}
        
        # Extract forehead region (typically clearest skin area)
        h, w = face_region.shape[:2]
        forehead = face_region[int(h*0.1):int(h*0.3), int(w*0.3):int(w*0.7)]
        
        if forehead.size == 0:
            return {"passed": False, "score": 0.0, "reason": "Cannot extract skin patch"}
        
        # Convert to grayscale
        gray = cv2.cvtColor(forehead, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (texture measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Real skin: variance > 20
        # Printed photo: variance < 10
        # Screen: variance 10-20
        
        if laplacian_var > 20:
            score = min(laplacian_var / 50, 1.0)
            passed = True
            reason = "Natural texture"
        elif laplacian_var > 10:
            score = 0.5
            passed = False
            reason = "Suspicious texture (screen?)"
        else:
            score = 0.2
            passed = False
            reason = "Flat texture (photo?)"
        
        return {
            "passed": passed,
            "score": score,
            "reason": reason,
            "variance": float(laplacian_var)
        }


class HeadMovementTracker:
    """Tracks micro-movements to detect static images"""
    
    def __init__(self):
        self.nose_positions = []
        self.max_history = 45  # 1.5 seconds at 30fps
        
    def track_movement(self, nose_landmark: Tuple[float, float]) -> Dict[str, Any]:
        """
        Track nose position for micro-movements
        Real person: constant small movements (breathing, natural swayy)
        Photo/screen: perfectly static
        """
        self.nose_positions.append(nose_landmark)
        
        # Keep limited history
        if len(self.nose_positions) > self.max_history:
            self.nose_positions.pop(0)
        
        if len(self.nose_positions) < 15:  # Need some history
            return {
                "passed": True,  # Pass initially
                "score": 0.5,
                "reason": "Collecting movement data..."
            }
        
        # Calculate movement variance
        positions_array = np.array(self.nose_positions)
        x_variance = np.var(positions_array[:, 0])
        y_variance = np.var(positions_array[:, 1])
        total_variance = x_variance + y_variance
        
        # Real person: variance > 0.00005 (micro-movements)
        # Static photo: variance < 0.00001
        
        if total_variance > 0.00005:
            score = min(total_variance * 50000, 1.0)
            passed = True
            reason = "Natural movement"
        elif total_variance > 0.00001:
            score = 0.4
            passed = False
            reason = "Minimal movement (video?)"
        else:
            score = 0.1
            passed = False
            reason = "No movement (photo?)"
        
        return {
            "passed": passed,
            "score": score,
            "reason": reason,
            "variance": float(total_variance)
        }


class LightConsistencyChecker:
    """Detects screen replay by analyzing light patterns"""
    
    def check_lighting(self, face_region: np.ndarray) -> Dict[str, Any]:
        """
        Analyze lighting consistency
        Real face: varied, natural lighting
        Screen: uniform, possible glare/reflections
        """
        if face_region is None or face_region.size == 0:
            return {"passed": False, "score": 0.0, "reason": "Invalid face region"}
        
        # Convert to HSV
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2]
        
        # Calculate brightness variance
        brightness_variance = np.var(brightness)
        
        # Check for unnatural highlights (screen glare)
        very_bright = np.sum(brightness > 240)
        total_pixels = brightness.size
        highlight_ratio = very_bright / total_pixels
        
        # Real face: variance > 150, highlights < 5%
        # Screen: low variance OR high highlights
        
        if brightness_variance > 150 and highlight_ratio < 0.05:
            score = min(brightness_variance / 400, 1.0)
            passed = True
            reason = "Natural lighting"
        elif highlight_ratio > 0.1:
            score = 0.2
            passed = False
            reason = "Screen glare detected"
        else:
            score = 0.4
            passed = False
            reason = "Uniform lighting (screen?)"
        
        return {
            "passed": passed,
            "score": score,
            "reason": reason,
            "variance": float(brightness_variance),
            "highlights": float(highlight_ratio)
        }


class AntiSpoofingEngine:
    """Main anti-spoofing engine combining all checks"""
    
    def __init__(self):
        self.blink_detector = BlinkDetector()
        self.texture_analyzer = TextureAnalyzer()
        self.movement_tracker = HeadMovementTracker()
        self.light_checker = LightConsistencyChecker()
        self.start_time = time.time()
        
    def verify_liveness(self, frame: np.ndarray, landmarks, blink_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive liveness verification
        
        Args:
            frame: Full frame
            landmarks: MediaPipe face landmarks
            blink_data: Dict with left_ear, right_ear, is_blinking
            
        Returns:
            {
                "is_live": bool,
                "confidence": float,
                "checks": {...},
                "status": str,
                "reason": str
            }
        """
        current_time = time.time() - self.start_time
        
        # Run all checks
        checks = {}
        
        # 1. Enhanced blink detection
        if blink_data:
            checks["blink"] = self.blink_detector.analyze_blink(
                blink_data.get("left_ear", 0.3),
                blink_data.get("right_ear", 0.3),
                current_time
            )
        else:
            checks["blink"] = {"passed": False, "score": 0.0, "reason": "No blink data"}
        
        # Extract face region for texture and light analysis
        if landmarks:
            h, w = frame.shape[:2]
            
            # Get bounding box from landmarks
            x_coords = [lm.x * w for lm in landmarks.landmark]
            y_coords = [lm.y * h for lm in landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            face_region = frame[y_min:y_max, x_min:x_max]
            
            # 2. Texture analysis
            checks["texture"] = self.texture_analyzer.analyze_face_texture(face_region)
            
            # 3. Movement tracking (nose tip)
            nose = landmarks.landmark[1]  # Nose tip
            checks["movement"] = self.movement_tracker.track_movement((nose.x, nose.y))
            
            # 4. Light consistency
            checks["light"] = self.light_checker.check_lighting(face_region)
        else:
            checks["texture"] = {"passed": False, "score": 0.0, "reason": "No landmarks"}
            checks["movement"] = {"passed": False, "score": 0.0, "reason": "No landmarks"}
            checks["light"] = {"passed": False, "score": 0.0, "reason": "No landmarks"}
        
        # Calculate overall score (weighted)
        weights = {
            "blink": 0.25,
            "texture": 0.35,
            "movement": 0.25,
            "light": 0.15
        }
        
        overall_score = sum(checks[k]["score"] * weights[k] for k in checks)
        
        # Decision logic: require critical checks + good score
        texture_passed = checks["texture"]["passed"]
        movement_passed = checks["movement"]["passed"]
        
        # Must pass texture AND movement (critical)
        # OR pass all 3 non-blink checks with high score
        is_live = (texture_passed and movement_passed) or \
                  (texture_passed and checks["light"]["passed"] and overall_score > 0.75)
        
        # Generate status message
        if is_live:
            status = "[OK] Live Person"
            reason = "All checks passed"
        else:
            # Find main failure reason
            failed_checks = [k for k, v in checks.items() if not v["passed"]]
            if "texture" in failed_checks:
                reason = checks["texture"]["reason"]
                status = "[ERROR] Spoof Detected"
            elif "movement" in failed_checks:
                reason = checks["movement"]["reason"]
                status = "[ERROR] Spoof Detected"
            else:
                reason = "Multiple checks failed"
                status = "[WARN] Verification Failed"
        
        return {
            "is_live": is_live,
            "confidence": round(overall_score, 2),
            "checks": checks,
            "status": status,
            "reason": reason
        }
    
    def reset(self):
        """Reset trackers (e.g., when person leaves frame)"""
        self.blink_detector = BlinkDetector()
        self.movement_tracker = HeadMovementTracker()
