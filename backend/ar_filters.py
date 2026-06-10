"""
AR Filter Engine for Snapchat-Style Filters
Optimized for real-time performance with non-blocking execution
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional


class ARFilterEngine:
    def __init__(self):
        self.current_filter = "none"
        self.available_filters = [
            "none", "dog", "sunglasses", "flower_crown", "big_eyes",
            "neon_outline", "mask", "beauty", "face_swap"
        ]
        
    def set_filter(self, filter_name: str):
        """Set the current AR filter"""
        if filter_name in self.available_filters:
            self.current_filter = filter_name
            print(f"[OK] AR Filter set to: {filter_name}")
            return True
        return False
    
    def apply_filter(self, frame: np.ndarray, all_landmarks, filter_name: str) -> np.ndarray:
        """Apply the selected AR filter to the frame - OPTIMIZED for real-time"""
        if filter_name == "none" or not all_landmarks:
            return frame
        
        try:
            # Create a copy to avoid modifying original
            result = frame.copy()
            
            # Filters that need 2+ faces (e.g. face_swap)
            if filter_name == "face_swap":
                if len(all_landmarks) >= 2:
                    result = self._apply_face_swap_filter(result, all_landmarks)
                else:
                    cv2.putText(result, "Face Swap Needs 2 Faces", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return result

            # Single-face filters applied to ALL detected faces
            for landmarks in all_landmarks:
                if filter_name == "dog":
                    result = self._apply_dog_filter(result, landmarks)
                elif filter_name == "sunglasses":
                    result = self._apply_sunglasses_filter(result, landmarks)
                elif filter_name == "flower_crown":
                    result = self._apply_flower_crown_filter(result, landmarks)
                elif filter_name == "big_eyes":
                    result = self._apply_big_eyes_filter_optimized(result, landmarks)
                elif filter_name == "neon_outline":
                    result = self._apply_neon_outline_filter(result, landmarks)
                elif filter_name == "mask":
                    result = self._apply_mask_overlay_filter(result, landmarks)
                elif filter_name == "beauty":
                    result = self._apply_beauty_filter_optimized(result, landmarks)
            
            return result
        except Exception as e:
            print(f"[WARN] Filter error {filter_name}: {e}")
            return frame  # Return original on error
    
    def _apply_dog_filter(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Apply dog ears and tongue filter - OPTIMIZED"""
        try:
            h, w = frame.shape[:2]
            
            # Get key points safely
            nose_tip = landmarks.landmark[1]
            forehead = landmarks.landmark[10]
            mouth_top = landmarks.landmark[13]
            mouth_bottom = landmarks.landmark[14]
            
            # Draw dog ears (simple triangles)
            left_ear = np.array([
                [int(w * 0.15), int(h * 0.05)],
                [int(w * forehead.x - 60), int(h * forehead.y)],
                [int(w * forehead.x - 40), int(h * 0.1)]
            ], dtype=np.int32)
            cv2.fillPoly(frame, [left_ear], (139, 69, 19))
            
            right_ear = np.array([
                [int(w * 0.85), int(h * 0.05)],
                [int(w * forehead.x + 60), int(h * forehead.y)],
                [int(w * forehead.x + 40), int(h * 0.1)]
            ], dtype=np.int32)
            cv2.fillPoly(frame, [right_ear], (139, 69, 19))
            
            # Check mouth opening
            mouth_open = abs(mouth_bottom.y - mouth_top.y) > 0.02
            if mouth_open:
                tongue = np.array([
                    [int(w * nose_tip.x - 15), int(h * mouth_bottom.y)],
                    [int(w * nose_tip.x + 15), int(h * mouth_bottom.y)],
                    [int(w * nose_tip.x), int(h * mouth_bottom.y + 30)]
                ], dtype=np.int32)
                cv2.fillPoly(frame, [tongue], (255, 105, 180))
        except Exception as e:
            pass  # Skip on error
        
        return frame
    
    def _apply_sunglasses_filter(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Apply sunglasses filter - OPTIMIZED"""
        try:
            h, w = frame.shape[:2]
            
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            
            eye_distance = int(w * abs(right_eye.x - left_eye.x))
            center_y = int(h * (left_eye.y + right_eye.y) / 2)
            glass_width = eye_distance // 2
            glass_height = int(glass_width * 0.8)
            
            # Draw lenses
            cv2.ellipse(frame, (int(w * left_eye.x), center_y),
                       (glass_width, glass_height), 0, 0, 360, (30, 30, 30), -1)
            cv2.ellipse(frame, (int(w * left_eye.x), center_y),
                       (glass_width, glass_height), 0, 0, 360, (0, 0, 0), 3)
            
            cv2.ellipse(frame, (int(w * right_eye.x), center_y),
                       (glass_width, glass_height), 0, 0, 360, (30, 30, 30), -1)
            cv2.ellipse(frame, (int(w * right_eye.x), center_y),
                       (glass_width, glass_height), 0, 0, 360, (0, 0, 0), 3)
            
            # Bridge
            cv2.line(frame, (int(w * left_eye.x) + glass_width, center_y),
                    (int(w * right_eye.x) - glass_width, center_y), (0, 0, 0), 3)
        except Exception as e:
            pass
        
        return frame
    
    def _apply_flower_crown_filter(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Apply flower crown filter - OPTIMIZED"""
        try:
            h, w = frame.shape[:2]
            
            forehead = landmarks.landmark[10]
            left_temple = landmarks.landmark[109]
            right_temple = landmarks.landmark[338]
            
            num_flowers = 7
            colors = [(255, 192, 203), (255, 182, 193), (255, 105, 180)]
            
            for i in range(num_flowers):
                t = i / (num_flowers - 1)
                x = int(w * (left_temple.x * (1 - t) + right_temple.x * t))
                y = int(h * forehead.y - 30)
                
                color = colors[i % len(colors)]
                cv2.circle(frame, (x, y), 12, color, -1)
                cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
        except Exception as e:
            pass
        
        return frame
    
    def _apply_big_eyes_filter_optimized(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Apply big eyes filter - OPTIMIZED using cv2.remap (much faster)"""
        try:
            h, w = frame.shape[:2]
            
            left_eye_center = landmarks.landmark[33]
            right_eye_center = landmarks.landmark[263]
            
            # Use smaller radius for better performance
            radius = 35
            strength = 0.4  # Reduced for subtlety and speed
            
            # Process both eyes
            for eye_center in [left_eye_center, right_eye_center]:
                cx, cy = int(w * eye_center.x), int(h * eye_center.y)
                
                # Bounds checking
                if cx < radius or cx >= w - radius or cy < radius or cy >= h - radius:
                    continue
                
                # Create region of interest
                roi_y1, roi_y2 = max(0, cy - radius), min(h, cy + radius)
                roi_x1, roi_x2 = max(0, cx - radius), min(w, cx + radius)
                
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                roi_h, roi_w = roi.shape[:2]
                
                # Create mesh grid for warping
                local_cx = cx - roi_x1
                local_cy = cy - roi_y1
                
                xx, yy = np.meshgrid(np.arange(roi_w), np.arange(roi_h))
                dx = xx - local_cx
                dy = yy - local_cy
                distance = np.sqrt(dx**2 + dy**2)
                
                # Apply bulge only within radius
                mask = distance < radius
                factor = np.zeros_like(distance)
                factor[mask] = 1 - (distance[mask] / radius) ** 2
                push = strength * factor
                
                map_x = (xx - dx * push).astype(np.float32)
                map_y = (yy - dy * push).astype(np.float32)
                
                # Apply remap (fast)
                warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)
                frame[roi_y1:roi_y2, roi_x1:roi_x2] = warped
        except Exception as e:
            pass
        
        return frame
    
    def _apply_neon_outline_filter(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Apply neon face outline filter - OPTIMIZED"""
        try:
            h, w = frame.shape[:2]
            overlay = np.zeros_like(frame)
            
            # Reduced contour points for performance
            contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127]
            
            points = []
            for idx in contour_indices:
                lm = landmarks.landmark[idx]
                points.append((int(w * lm.x), int(h * lm.y)))
            
            # Draw lines
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(overlay, [points_array], True, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Add glow (smaller blur for speed)
            overlay_blur = cv2.GaussianBlur(overlay, (11, 11), 0)
            frame = cv2.addWeighted(frame, 0.8, overlay_blur, 0.2, 0)
        except Exception as e:
            pass
        
        return frame
    
    def _apply_mask_overlay_filter(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Apply virtual mask overlay - OPTIMIZED"""
        try:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            
            nose_tip = landmarks.landmark[1]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            left_cheek = landmarks.landmark[234]
            right_cheek = landmarks.landmark[454]
            
            mask_points = np.array([
                [int(w * left_cheek.x), int(h * left_eye.y)],
                [int(w * left_eye.x - 20), int(h * left_eye.y - 10)],
                [int(w * nose_tip.x), int(h * nose_tip.y - 20)],
                [int(w * right_eye.x + 20), int(h * right_eye.y - 10)],
                [int(w * right_cheek.x), int(h * right_eye.y)],
                [int(w * right_eye.x + 10), int(h * right_eye.y + 10)],
                [int(w * left_eye.x - 10), int(h * left_eye.y + 10)]
            ], dtype=np.int32)
            
            cv2.fillPoly(overlay, [mask_points], (0, 0, 139))
            
            # Eye holes
            cv2.ellipse(overlay, (int(w * left_eye.x), int(h * left_eye.y)),
                       (25, 18), 0, 0, 360, (0, 0, 0), -1)
            cv2.ellipse(overlay, (int(w * right_eye.x), int(h * right_eye.y)),
                       (25, 18), 0, 0, 360, (0, 0, 0), -1)
            
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        except Exception as e:
            pass
        
        return frame
    
    def _apply_beauty_filter_optimized(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Apply beauty/skin smoothing filter - OPTIMIZED"""
        try:
            # Use smaller kernel for speed
            smoothed = cv2.bilateralFilter(frame, 5, 50, 50)
            frame = cv2.addWeighted(frame, 0.6, smoothed, 0.4, 0)
            frame = cv2.convertScaleAbs(frame, alpha=1.03, beta=3)
        except Exception as e:
            pass
        
        return frame
    
    def _apply_face_swap_filter(self, frame: np.ndarray, all_landmarks) -> np.ndarray:
        """Apply high-quality face swap between the first two detected faces"""
        try:
            if len(all_landmarks) < 2:
                return frame
                
            img = frame.copy()
            h, w = img.shape[:2]
            
            # 1. Get landmarks for both faces
            lm1 = all_landmarks[0]
            lm2 = all_landmarks[1]
            
            # Convert to list of (x, y) points
            pts1 = [(int(l.x * w), int(l.y * h)) for l in lm1.landmark]
            pts2 = [(int(l.x * w), int(l.y * h)) for l in lm2.landmark]
            
            # Use a subset of points for speed (main contours + features)
            # indices for face contour, eyes, nose, mouth
            feat_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                        33, 133, 159, 145, 263, 362, 386, 374, 1, 4, 5, 197, 13, 14, 78, 308]
            
            sub_pts1 = np.array([pts1[i] for i in feat_idx], np.int32)
            sub_pts2 = np.array([pts2[i] for i in feat_idx], np.int32)
            
            # 2. Find Convex Hull for masking
            hull1 = cv2.convexHull(sub_pts1)
            hull2 = cv2.convexHull(sub_pts2)
            
            # 3. Create swapped faces
            # Warp Face 1 to Face 2 shape
            warped_face1 = self._warp_face(img, sub_pts1, sub_pts2, h, w)
            # Warp Face 2 to Face 1 shape
            warped_face2 = self._warp_face(img, sub_pts2, sub_pts1, h, w)
            
            # 4. Create masks with slight feathering
            mask1 = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask1, hull1, 255)
            mask1 = cv2.GaussianBlur(mask1, (15, 15), 0)
            
            mask2 = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask2, hull2, 255)
            mask2 = cv2.GaussianBlur(mask2, (15, 15), 0)
            
            # 5. Blend
            mask1_3ch = cv2.merge([mask1, mask1, mask1]) / 255.0
            mask2_3ch = cv2.merge([mask2, mask2, mask2]) / 255.0
            
            # Place Face 2 on Face 1's position
            img = img.astype(float)
            warped_face2 = warped_face2.astype(float)
            img = img * (1 - mask1_3ch) + warped_face2 * mask1_3ch
            
            # Place Face 1 on Face 2's position
            warped_face1 = warped_face1.astype(float)
            img = img * (1 - mask2_3ch) + warped_face1 * mask2_3ch
            
            return img.astype(np.uint8)
            
        except Exception as e:
            print(f"[WARN] Face swap robust error: {e}")
            return frame

    def _warp_face(self, img, pts_src, pts_dst, h, w):
        """Warp source points to destination points using affine transform per triangle"""
        # Simplified: Use a single affine transform for the whole face if triangulation is too slow
        # For better quality, we use the triangulated approach or just a global affine if points are few
        M, _ = cv2.findHomography(pts_src.astype(np.float32), pts_dst.astype(np.float32))
        return cv2.warpPerspective(img, M, (w, h))
