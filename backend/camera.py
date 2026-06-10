"""
Camera Manager for handling video capture from webcam or file
Runs in a separate thread for non-blocking frame capture
"""
import cv2
import threading
import time
from typing import Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.active = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        
    def start(self, source=0) -> bool:
        """
        Start camera capture
        source: 0 for webcam, or path to video file
        """
        logger.info(f"Starting camera with source: {source}")
        
        if self.active:
            logger.warning("Camera already active, stopping first...")
            self.stop()
        
        try:
            if isinstance(source, int):
                logger.info(f"Opening camera device {source} with DirectShow")
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                logger.info(f"Opening video file: {source}")
                self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {source}")
                logger.error("Possible issues: Camera permission denied, device busy, or invalid source")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.active = True
            self.start_time = time.time()
            self.frame_count = 0
            
            # Start capture thread
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info("Camera capture thread started successfully")
            
            return True
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            return False
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.active:
            ret, frame = self.cap.read()
            
            if ret:
                consecutive_failures = 0
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
                    
                    # Calculate FPS
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
                        
                    # Log FPS periodically
                    if self.frame_count % 100 == 0:
                        logger.info(f"Camera FPS: {self.fps:.1f}, Frames captured: {self.frame_count}")
            else:
                consecutive_failures += 1
                logger.warning(f"Frame read failed (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive frame read failures, stopping camera")
                    self.active = False
                    break
                    
                # Brief pause before retry
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def stop(self):
        """Stop camera capture"""
        logger.info("Stopping camera...")
        self.active = False
        
        if self.thread is not None:
            logger.info("Waiting for capture thread to finish...")
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Capture thread did not finish cleanly")
        
        if self.cap is not None:
            logger.info("Releasing camera device")
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.frame = None
        
        logger.info("Camera stopped successfully")
    
    def is_active(self) -> bool:
        """Check if camera is active"""
        return self.active
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return round(self.fps, 2)
    
    def get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
