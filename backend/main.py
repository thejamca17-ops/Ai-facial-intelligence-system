import os
import logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to avoid GPU init issues

from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import cv2
import asyncio
import time
import json
import base64
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from camera import CameraManager
from processor import FrameProcessor
from reporting import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")

app = FastAPI(title="AI Facial Intelligence System")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
camera_manager = CameraManager()
frame_processor = FrameProcessor()
report_generator = ReportGenerator()

# Session Data
session_logs: List[Dict] = []
face_logs: List[Dict] = []      # Enriched per-face biometric snapshots
session_start_time: float = 0.0  # Epoch — set when camera starts

# Live Registration Session State
live_registration_session = {
    "active": False,
    "samples": [],
    "target_samples": 26,
    "person_name": None,
    "person_gender": None
}

@app.get("/")
async def root():
    return {"message": "AI Facial Intelligence System API", "status": "running"}

@app.get("/health")
async def health_check():
    dataset_stats = {}
    if frame_processor.face_recognition_engine:
        dataset_stats = frame_processor.face_recognition_engine.get_database_stats()
    return {
        "status": "healthy",
        "camera_active": camera_manager.is_active(),
        "enabled_features": frame_processor.get_enabled_features(),
        "dataset": dataset_stats
    }

@app.post("/camera/start")
async def start_camera(source: int = 0):
    """Start camera capture"""
    global session_start_time
    success = camera_manager.start(source)
    if success:
        session_start_time = time.time()
    return {"success": success, "message": "Camera started" if success else "Failed to start camera"}

@app.post("/camera/stop")
async def stop_camera():
    """Stop camera capture"""
    camera_manager.stop()
    return {"success": True, "message": "Camera stopped"}

@app.post("/features/toggle")
async def toggle_feature(feature: str, enabled: bool):
    """Toggle AI feature on/off. When enabling, auto-enables required dependencies."""
    auto_enabled = []

    if enabled:
        # Auto-enable dependencies recursively so features actually work
        def _enable_deps(feat):
            for dep in frame_processor.feature_dependencies.get(feat, []):
                if not frame_processor.features.get(dep, False):
                    frame_processor.toggle_feature(dep, True)
                    auto_enabled.append(dep)
                    _enable_deps(dep)  # recurse for nested deps
        _enable_deps(feature)

    success = frame_processor.toggle_feature(feature, enabled)
    logger.info(f"[FEATURE] '{feature}' -> {enabled}. Auto-enabled deps: {auto_enabled}")
    return {
        "feature": feature,
        "enabled": enabled,
        "success": success,
        "auto_enabled_dependencies": auto_enabled,
        "all_features": frame_processor.get_all_features()  # send full state so frontend syncs
    }

@app.get("/features/list")
async def list_features():
    """Get all available features and their status"""
    return frame_processor.get_all_features()

@app.post("/ar-filter/set")
async def set_ar_filter(filter_name: str = Form(...)):
    """Set the current AR filter"""
    try:
        if frame_processor.ar_filter_engine:
            success = frame_processor.ar_filter_engine.set_filter(filter_name)
            if success:
                return {
                    "success": True,
                    "filter": filter_name,
                    "message": f"AR Filter set to: {filter_name}"
                }
            else:
                return JSONResponse(
                    {"error": f"Invalid filter. Available: {frame_processor.ar_filter_engine.available_filters}"},
                    status_code=400
                )
        else:
            return JSONResponse({"error": "AR Filter Engine not initialized"}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/ar-filter/list")
async def list_ar_filters():
    """Get list of available AR filters"""
    if frame_processor.ar_filter_engine:
        return {
            "current_filter": frame_processor.ar_filter_engine.current_filter,
            "available_filters": frame_processor.ar_filter_engine.available_filters
        }
    else:
        return JSONResponse({"error": "AR Filter Engine not initialized"}, status_code=503)

@app.post("/report/export")
async def export_report():
    """Generate and return PDF report (non-blocking)"""
    import asyncio

    # Build data payload — works even when logs are empty
    now_ts = time.time()
    start_str = (
        datetime.fromtimestamp(session_start_time).strftime("%Y-%m-%d %H:%M:%S")
        if session_start_time else "—"
    )
    end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "total_frames": camera_manager.frame_count,
        "avg_fps": camera_manager.get_fps(),
        "session_start": start_str,
        "session_end": end_str,
        "logs": list(session_logs),           # basic detection log
        "face_logs": list(face_logs),         # enriched per-face biometric snapshots
    }

    try:
        # Run in thread pool so camera WebSocket is not blocked
        loop = asyncio.get_event_loop()
        pdf_path = await loop.run_in_executor(None, report_generator.create_report, data)
        return FileResponse(
            pdf_path,
            filename=f"facial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            media_type="application/pdf",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# Face Recognition Endpoints

@app.post("/recognition/register")
async def register_person(name: str = Form(...), gender: str = Form("Unknown"), files: List[UploadFile] = File(...)):
    """Register a new person with face images and gender (batch upload)"""
    logger.info(f"[REG] Batch registration request: name='{name}' gender='{gender}' files={len(files)}")
    try:
        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition not initialized"}, status_code=503)

        images_list = []
        for file in files:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images_list.append(img)
            else:
                logger.warning(f"[REG] Could not decode uploaded file: {file.filename}")

        if not images_list:
            return JSONResponse({"error": "No valid images could be decoded from the provided files"}, status_code=400)

        loop = asyncio.get_event_loop()
        try:
            success = await loop.run_in_executor(
                None,
                lambda: frame_processor.face_recognition_engine.register_new_person(name, images_list, gender)
            )
        except ValueError as ve:
            return JSONResponse({"error": str(ve)}, status_code=400)

        logger.info(f"[REG] Registration result: name='{name}' success={success}")
        return {
            "success": success,
            "message": f"Registered '{name}' ({gender}) with {len(images_list)} images" if success else "Registration failed",
            "name": name,
            "gender": gender,
            "images_count": len(images_list)
        }
    except Exception as e:
        logger.exception(f"[REG] Unexpected error during registration of '{name}'")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/reset")
async def reset_database():
    """Reset all registered persons and their dataset images"""
    try:
        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition not initialized"}, status_code=503)
            
        success = frame_processor.face_recognition_engine.reset_database()
        
        return {
            "success": success,
            "message": "All registered persons and dataset images have been reset successfully"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/delete")
async def delete_person(name: str = Form(...)):
    """Delete a specific person from the recognition system (non-blocking)"""
    logger.info(f"[DEL] Delete request for person: '{name}'")
    try:
        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition not initialized"}, status_code=503)

        person_name = (name or "").strip()
        if not person_name:
            return JSONResponse({"error": "Name is required"}, status_code=400)

        # Run in thread pool so the camera WebSocket stays responsive
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            frame_processor.face_recognition_engine.delete_person,
            person_name
        )

        if result.get("success"):
            logger.info(f"[DEL] Successfully deleted '{person_name}' ({result.get('removed_count', 0)} embeddings)")
        else:
            logger.warning(f"[DEL] Delete failed for '{person_name}': {result.get('error')}")

        # Always return 200 with payload so frontend can check result.success
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        logger.exception(f"[DEL] Unexpected error deleting '{name}'")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/rebuild")
async def rebuild_database(dataset_path: str = Form("default")):
    """Rebuild embeddings from dataset folder"""
    try:
        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition not initialized"}, status_code=503)
        
        # Use default path if requested
        actual_path = dataset_path
        if dataset_path == "default" or not dataset_path:
            actual_path = os.path.join(os.path.dirname(__file__), "dataset")
            
        if not os.path.exists(actual_path):
            return JSONResponse({"error": f"Dataset path does not exist: {actual_path}"}, status_code=400)

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: frame_processor.face_recognition_engine.build_embeddings_database(
                actual_path,
                frame_processor.face_recognition_engine.database_path
            )
        )
        
        stats = frame_processor.face_recognition_engine.get_database_stats()
        
        return {
            "success": success,
            "message": "Dataset rebuilt successfully" if success else "Rebuild failed",
            "stats": stats
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/rescan")
async def rescan_dataset():
    """
    Re-scan the dataset folder and reload all embeddings into memory.
    Use this after manually adding images to the dataset folder,
    or after fixing/renaming person folders, without restarting the server.
    """
    logger.info("[RESCAN] Re-scanning dataset folder...")
    try:
        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition not initialized"}, status_code=503)

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            frame_processor.face_recognition_engine.rescan_dataset
        )

        logger.info(f"[RESCAN] Done: {stats}")
        return {
            "success": True,
            "message": "Dataset re-scanned successfully",
            "stats": stats
        }
    except Exception as e:
        logger.exception("[RESCAN] Failed")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/live/start")
async def start_live_registration():
    """Start a live registration session"""
    try:
        if live_registration_session["active"]:
            return JSONResponse({"error": "A registration session is already active"}, status_code=400)
        
        live_registration_session["active"] = True
        live_registration_session["samples"] = []
        live_registration_session["person_name"] = None
        live_registration_session["person_gender"] = None
        
        return {
            "success": True,
            "message": "Live registration session started",
            "target_samples": live_registration_session["target_samples"]
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/live/capture")
async def add_live_sample(file: UploadFile = File(...)):
    """Add a face sample to the current registration session"""
    try:
        if not live_registration_session["active"]:
            return JSONResponse({"error": "No active registration session"}, status_code=400)
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
        
        # Store sample
        live_registration_session["samples"].append(img)
        current_count = len(live_registration_session["samples"])
        
        return {
            "success": True,
            "samples_collected": current_count,
            "target_samples": live_registration_session["target_samples"],
            "complete": current_count >= live_registration_session["target_samples"]
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/live/complete")
async def complete_live_registration(name: str = Form(...), gender: str = Form(...)):
    """Complete registration with collected live-camera samples"""
    logger.info(f"[LIVE-REG] Complete request: name='{name}' gender='{gender}'")
    try:
        if not live_registration_session["active"]:
            return JSONResponse({"error": "No active registration session. Please start a new session first."}, status_code=400)

        samples = live_registration_session.get("samples", [])
        target = live_registration_session.get("target_samples", 25)
        
        if not samples:
            return JSONResponse({"error": "No face samples were collected. Please restart the registration."}, status_code=400)

        if len(samples) < target:
            logger.warning(f"[LIVE-REG] Attempted registration with only {len(samples)}/{target} samples.")
            return JSONResponse({
                "error": f"Minimum {target} samples required. Collected only {len(samples)}.",
                "collected": len(samples),
                "required": target
            }, status_code=400)

        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition engine is not initialized"}, status_code=503)

        samples_count = len(samples)
        logger.info(f"[LIVE-REG] Processing {samples_count} samples for '{name}'")

        # Register in background thread so the camera WebSocket stays live
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                frame_processor.face_recognition_engine.register_new_person,
                name,
                list(samples),  # pass a copy
                gender
            )
        except ValueError as ve:
            logger.warning(f"[LIVE-REG] Validation error for '{name}': {ve}")
            live_registration_session["active"] = False
            live_registration_session["samples"] = []
            return JSONResponse({"error": str(ve)}, status_code=400)
        except Exception as e:
            logger.exception(f"[LIVE-REG] Registration failed for '{name}'")
            live_registration_session["active"] = False
            live_registration_session["samples"] = []
            return JSONResponse({"error": f"Registration failed: {str(e)}"}, status_code=500)
        finally:
            # Always reset session state
            live_registration_session["active"] = False
            live_registration_session["samples"] = []
            live_registration_session["person_name"] = None
            live_registration_session["person_gender"] = None

        # Verify the person was actually persisted
        persisted = frame_processor.face_recognition_engine._verify_person_in_db(name)
        if not persisted:
            logger.error(f"[LIVE-REG] Persistence check failed for '{name}'")
            return JSONResponse({"error": "Registration appeared to succeed but the person could not be verified in the dataset. Please try again."}, status_code=500)

        logger.info(f"[LIVE-REG] Successfully registered '{name}' with {samples_count} samples")
        return {
            "success": True,
            "message": f"Successfully registered '{name}' with {samples_count} samples",
            "name": name,
            "gender": gender,
            "samples_used": samples_count
        }
    except Exception as e:
        logger.exception(f"[LIVE-REG] Unexpected error for '{name}'")
        live_registration_session["active"] = False
        live_registration_session["samples"] = []
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/recognition/live/cancel")
async def cancel_live_registration():
    """Cancel the current registration session (idempotent — safe to call even if already complete)"""
    try:
        if not live_registration_session["active"]:
            # Session may have ended via /complete — return OK, not an error
            return {"success": True, "message": "No active session (already completed or not started)"}
        
        samples_count = len(live_registration_session["samples"])
        
        # Reset session
        live_registration_session["active"] = False
        live_registration_session["samples"] = []
        live_registration_session["person_name"] = None
        live_registration_session["person_gender"] = None
        
        return {
            "success": True,
            "message": f"Registration cancelled ({samples_count} samples discarded)"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/recognition/persons")
async def get_registered_persons():
    """Get list of all registered persons"""
    try:
        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition not initialized"}, status_code=503)

        loop = asyncio.get_event_loop()
        persons = await loop.run_in_executor(
            None,
            frame_processor.face_recognition_engine.get_registered_persons
        )
        logger.info(f"[PERSONS] Fetched {len(persons)} registered persons")
        return {"success": True, "persons": persons}
    except Exception as e:
        logger.exception("[PERSONS] Failed to fetch persons list")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/recognition/stats")
async def get_recognition_stats():
    """Get face recognition statistics"""
    try:
        if not frame_processor.face_recognition_engine:
            return JSONResponse({"error": "Face recognition not initialized"}, status_code=503)

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            None,
            frame_processor.face_recognition_engine.get_database_stats
        )
        return stats
    except Exception as e:
        logger.exception("[STATS] Failed to fetch stats")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming with AI processing"""
    await websocket.accept()
    
    try:
        while True:
            if camera_manager.is_active():
                frame = camera_manager.get_frame()
                
                if frame is not None:
                    try:
                        # Process frame with AI
                        processed_frame, attributes = frame_processor.process_frame(frame)
                        
                        if processed_frame is None or attributes is None:
                            print("Warning: Frame processing returned None for processed_frame or attributes. Skipping frame.")
                            await asyncio.sleep(0.033) # Still wait to prevent busy loop
                            continue

                        # Encode frame to JPEG
                        _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Log data — summary + per-face biometrics
                        if "faces" in attributes and len(attributes["faces"]) > 0:
                            ts_str = datetime.fromtimestamp(attributes["timestamp"]).strftime("%H:%M:%S")
                            session_logs.append({
                                "time": ts_str,
                                "face_count": len(attributes["faces"]),
                                "attributes": "Faces Detected"
                            })
                            # Capture per-face biometric snapshot (sampled every ~2s)
                            if len(face_logs) == 0 or (time.time() - face_logs[-1].get("_ts", 0) > 2.0):
                                for face in attributes["faces"]:
                                    entry = {
                                        "_ts": time.time(),
                                        "time": ts_str,
                                        "face_id": face.get("face_id") or face.get("id"),
                                        "name": face.get("name", "Unknown"),
                                        "recognition_confidence": face.get("recognition_confidence", 0),
                                        "recognition_locked": face.get("recognition_locked", False),
                                        "registered_gender": face.get("registered_gender", "Unknown"),
                                        "emotion": face.get("emotion", ""),
                                        "stress_level": (face.get("stress") or {}).get("level", ""),
                                        "blink_count": (face.get("blink") or {}).get("count", 0),
                                        "attention_level": (face.get("attention") or {}).get("level", ""),
                                        "anti_spoof_status": (face.get("anti_spoof") or {}).get("status", ""),
                                        "anti_spoof_live": (face.get("anti_spoof") or {}).get("is_live"),
                                        "anti_spoof_reason": (face.get("anti_spoof") or {}).get("reason", ""),
                                    }
                                    face_logs.append(entry)
                        
                        # Send frame and attributes
                        await websocket.send_json({
                            "frame": frame_base64,
                            "attributes": attributes,
                            "timestamp": camera_manager.get_timestamp()
                        })
                    except Exception as processing_e:
                        print(f"Error during frame processing: {processing_e}")
                        import traceback
                        traceback.print_exc()
                        # FALLBACK: Send raw frame so video feed still works
                        try:
                            _, raw_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            raw_base64 = base64.b64encode(raw_buffer).decode('utf-8')
                            await websocket.send_json({
                                "frame": raw_base64,
                                "attributes": {"faces": [], "fps": 0, "timestamp": time.time(), "face_count": 0, "error": str(processing_e)},
                                "timestamp": camera_manager.get_timestamp()
                            })
                        except Exception:
                            pass
                
                await asyncio.sleep(0.016)  # ~60 FPS cap to keep event loop responsive
            else:
                await asyncio.sleep(0.1)
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass # Connection already closed

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Process uploaded image"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
        
        processed_frame, attributes = frame_processor.process_frame(frame)
        
        # Encode processed frame
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "processed_image": frame_base64,
            "attributes": attributes
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Process uploaded video"""
    try:
        # Save uploaded video temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process video
        results = frame_processor.process_video(temp_path)
        
        return {
            "success": True,
            "results": results,
            "total_frames": len(results)
        }                                                                                            
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
