"""
Face Recognition Engine
Handles face dataset management, embedding extraction, and face matching.
Uses DeepFace with Facenet512 for high-accuracy recognition.

All persistence is folder-based:
  dataset/
    THEJA/
      img0.jpg
      img1.jpg
      gender.txt
    SATHI/
      img0.jpg
      ...

No database (SQLite, PostgreSQL, FAISS, or JSON) is used.
"""
import os
import cv2
import numpy as np
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
import threading

logger = logging.getLogger(__name__)

# Module-level DeepFace import cache — avoids re-importing on every call
_deepface_module = None
_deepface_lock   = threading.Lock()

def _get_deepface():
    """Return the DeepFace module, importing it once and caching it."""
    global _deepface_module
    if _deepface_module is None:
        with _deepface_lock:
            if _deepface_module is None:
                from deepface import DeepFace
                _deepface_module = DeepFace
    return _deepface_module


class FaceRecognitionEngine:
    def __init__(self, model: str = 'Facenet512', threshold: float = 0.75,
                 dataset_root: str = ''):
        """
        Initialize Face Recognition Engine using the Dataset Folder Method.

        Args:
            model:        DeepFace model name (Facenet512, VGG-Face, ArcFace, …)
            threshold:    Cosine-similarity threshold for recognition (0-1)
            dataset_root: Root path of the dataset folder (person sub-folders with images)
        """
        self.model     = model
        self.threshold = threshold

        # Duplicate detection threshold (higher = stricter)
        self.duplicate_threshold = 0.80

        # Resolve dataset root — default to <backend>/dataset
        if dataset_root:
            self.dataset_root = Path(dataset_root)
        else:
            self.dataset_root = Path(os.path.dirname(os.path.abspath(__file__))) / "dataset"
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        # Keep legacy attribute for any callers that reference database_path
        self.database_path = str(self.dataset_root.parent)

        # In-memory embedding store: list of dicts
        # Each entry: {name, embedding, gender, source_image, dataset_path}
        self.registered_faces: List[Dict[str, Any]] = []

        # Thread safety
        self.lock = threading.Lock()

        # Build embeddings from dataset on startup
        self._scan_dataset()

        print(f"[OK] Face Recognition Engine ready - model={self.model}, "
              f"threshold={self.threshold}, "
              f"embeddings={len(self.registered_faces)}")

    # ------------------------------------------------------------------
    # Dataset scanning
    # ------------------------------------------------------------------
    def _scan_dataset(self):
        """
        Scan the dataset folder and extract embeddings for every person.
        Called on startup. Only loads persons whose folder still exists.
        """
        if not self.dataset_root.exists():
            print("[INFO] Dataset folder not found — starting with empty store.")
            return

        persons = [p for p in self.dataset_root.iterdir() if p.is_dir()]
        if not persons:
            print("[INFO] No person folders in dataset — starting with empty store.")
            return

        print(f"[SCAN] Scanning dataset: {self.dataset_root} ({len(persons)} persons)")
        total_added = 0

        for person_dir in sorted(persons):
            person_name = person_dir.name
            image_files = list(self._find_images(person_dir))

            if not image_files:
                continue

            gender = self._read_gender(person_dir)

            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    emb = self._extract_embedding_from_stored_image(img)
                    if emb is not None:
                        self.registered_faces.append({
                            "name": person_name,
                            "embedding": emb,
                            "gender": gender,
                            "source_image": str(img_path),
                            "dataset_path": str(person_dir)
                        })
                        total_added += 1
                        print(f"  [SCAN] OK {person_name}/{img_path.name}")
                    else:
                        print(f"  [WARN] No face found in {img_path.name} — skipping")
                except Exception as e:
                    print(f"[WARN] Could not process {img_path}: {e}")

        print(f"[SCAN] Done: {total_added} embeddings loaded from {len(persons)} persons")

    def _extract_embedding_from_stored_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from a stored dataset image.
        Uses face detection (skip_detection=False) so full-frame images work correctly.
        Falls back to skip_detection=True if no face is found.
        """
        # Try with face detection first (for full-frame images)
        emb = self.extract_embedding(img, skip_detection=False)
        if emb is not None:
            return emb
        # Fallback: skip detection (for already-cropped face images)
        print("[SCAN] Face not found with detector — trying direct embedding (cropped image?)")
        return self.extract_embedding(img, skip_detection=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_images(self, folder: Path) -> List[Path]:
        """Return all image files inside a folder."""
        images = []
        # Case-insensitive extensions, deduplicated because of Windows behavior
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        all_files = list(folder.iterdir())
        for f in all_files:
            if f.suffix.lower() in exts:
                images.append(f)
        return sorted(list(set(images)))

    def _read_gender(self, person_dir: Path) -> str:
        """Read gender from gender.txt in a person folder, default Unknown."""
        meta_file = person_dir / "gender.txt"
        if meta_file.exists():
            try:
                return meta_file.read_text().strip()
            except Exception:
                pass
        return "Unknown"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = a_flat / (np.linalg.norm(a_flat) + 1e-10)
        norm_b = b_flat / (np.linalg.norm(b_flat) + 1e-10)
        return float(np.dot(norm_a, norm_b))

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------
    def extract_embedding(self, face_image: np.ndarray,
                          skip_detection: bool = False) -> Optional[np.ndarray]:
        """
        Extract a 512-d facial embedding from a face image.

        Args:
            face_image:      BGR image (from OpenCV)
            skip_detection:  If True, DeepFace skips its internal detector
        Returns:
            numpy array or None if extraction fails
        """
        try:
            DeepFace = _get_deepface()

            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image

            if skip_detection:
                face_rgb = cv2.resize(face_rgb, (160, 160))

            backend = 'skip' if skip_detection else 'opencv'
            embedding_objs = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.model,
                enforce_detection=False,
                detector_backend=backend,
                align=(not skip_detection)
            )

            if embedding_objs:
                return np.array(embedding_objs[0]["embedding"])

        except Exception as e:
            print(f"[WARN] Embedding extraction error: {e}")

        return None

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def recognize_face(self, face_image: np.ndarray, skip_detection: bool = False) -> Tuple[str, float, str]:
        """
        Identify a face against the registered dataset.

        Returns:
            (name, confidence_pct, gender) — ("Unknown", 0.0, "Unknown") if no match
        """
        query_embedding = self.extract_embedding(face_image, skip_detection=skip_detection)
        if query_embedding is None:
            return ("Unknown", 0.0, "Unknown")

        with self.lock:
            if not self.registered_faces:
                return ("Unknown", 0.0, "Unknown")

            best_name   = "Unknown"
            best_score  = 0.0
            best_gender = "Unknown"

            for face in self.registered_faces:
                score = self._cosine_similarity(query_embedding, face['embedding'])
                # Safe ASCII logging
                # print(f"[RECOG-DEBUG] Comparing to {face['name']}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_name = face['name']
                    best_gender = face.get('gender', 'Unknown')

            if best_score >= self.threshold:
                print(f"[RECOG] Match found: {best_name} ({best_score:.3f})")
                return (best_name, float(best_score * 100), best_gender)
            else:
                print(f"[RECOG] No Match. Best was {best_name} at {best_score:.3f} (thresh {self.threshold})")

        return ("Unknown", 0.0, "Unknown")

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------
    def check_duplicate_face(self, images_list: List[np.ndarray]) -> Optional[str]:
        """
        Check whether any face in images_list matches an already-registered person.

        Returns:
            The matched person's name if a duplicate is found, else None.
        """
        with self.lock:
            if not self.registered_faces:
                return None

        for img in images_list[:5]:  # Sample first few frames
            # Use detection/alignment for consistent duplicate check
            emb = self._extract_embedding_from_stored_image(img)
            if emb is None:
                continue
            with self.lock:
                for face in self.registered_faces:
                    score = self._cosine_similarity(emb, face['embedding'])
                    if score >= self.duplicate_threshold:
                        return face['name']
        return None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_new_person(self, name: str, images_list: List[np.ndarray],
                            gender: str = "Unknown") -> bool:
        """
        Register a new person by saving images into the dataset folder and
        adding their embeddings to the in-memory store.

        Raises ValueError if:
          - Name already exists (name-based check)
          - Face is too similar to an existing person (embedding-based check)
          - No valid embeddings could be extracted

        Returns True on success.
        """
        t0 = time.time()
        print(f"\n[REG] Registering: {name} ({gender}) - {len(images_list)} samples")

        # -- 1. Name-based duplicate check --------------------------------
        existing = self.get_registered_persons()
        if any(p['name'].lower() == name.strip().lower() for p in existing):
            raise ValueError(f"Person '{name}' already exists in the dataset!")

        # -- 2. Embedding-based duplicate check ---------------------------
        duplicate_match = self.check_duplicate_face(images_list)
        if duplicate_match:
            raise ValueError(
                "Person Already Exists - the face is too similar to "
                "Please delete the existing entry first."
            )

        # ── 3. Prepare folder ─────────────────────────────────────────────
        safe_name  = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        person_dir = self.dataset_root / safe_name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Save gender metadata
        try:
            (person_dir / "gender.txt").write_text(gender)
        except Exception:
            pass

        # ── 4. Deduplicate frames ────────────────────────────────────────
        unique_images = [images_list[0]]
        for img in images_list[1:]:
            if img.shape == unique_images[-1].shape:
                diff = np.mean(np.abs(img.astype(float) - unique_images[-1].astype(float)))
                if diff < 3.0:
                    continue
            unique_images.append(img)

        print(f"[REG] {len(unique_images)} unique frames after deduplication (was {len(images_list)})")

        # ── 5. Save images to disk ───────────────────────────────────────
        image_paths: List[str] = []
        for idx, img in enumerate(unique_images):
            path = str(person_dir / f"img{idx}.jpg")
            try:
                cv2.imwrite(path, img)
                image_paths.append(path)
            except Exception as e:
                print(f"[WARN] Could not save image {idx}: {e}")
                image_paths.append("")

        # ── 6. Extract embeddings in parallel ───────────────────────────
        results: Dict[int, Optional[np.ndarray]] = {}
        max_workers = min(4, len(unique_images))

        def _extract_one(args):
            idx, img = args
            try:
                # Use detection/alignment for registration to ensure high quality
                return idx, self._extract_embedding_from_stored_image(img)
            except Exception as ex:
                print(f"[WARN] extract_embedding failed for frame {idx}: {ex}")
                return idx, None

        print(f"[REG] Extracting embeddings ({max_workers} parallel workers)…")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_one, (i, img)): i
                       for i, img in enumerate(unique_images)}
            for future in as_completed(futures):
                idx, emb = future.result()
                results[idx] = emb

        # ── 7. Add valid embeddings to in-memory store ────────────────────
        added_count = 0
        with self.lock:
            for idx in range(len(unique_images)):
                emb = results.get(idx)
                if emb is not None:
                    self.registered_faces.append({
                        "name": name,
                        "embedding": emb,
                        "gender": gender,
                        "source_image": image_paths[idx] if idx < len(image_paths) else "",
                        "dataset_path": str(person_dir)
                    })
                    added_count += 1
                else:
                    print(f"  [SKIP] frame {idx}: no face detected / no embedding")

        if added_count == 0:
            # Clean up the created folder if nothing was registered
            try:
                shutil.rmtree(person_dir, ignore_errors=True)
            except Exception:
                pass
            raise ValueError("No valid face embeddings could be extracted from the provided images.")

        # ── 8. Self-recognition verification ────────────────────────────
        first_valid = next((i for i in range(len(unique_images)) if results.get(i) is not None), None)
        if first_valid is not None:
            test_name, test_conf, _ = self.recognize_face(unique_images[first_valid])
            if test_name.lower() == name.lower():
                print(f"  [OK] Self-recognition passed ({test_conf:.1f}%)")
            else:
                print(f"  [WARN] Self-recognition failed! Got '{test_name}' instead of '{name}'.")

        elapsed = time.time() - t0
        print(f"[OK] Registered '{name}': {added_count}/{len(unique_images)} embeddings in {elapsed:.1f}s")
        return True

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def delete_person(self, person_name: str) -> Dict[str, Any]:
        """
        Delete a person from:
          1. The in-memory embedding store
          2. The dataset folder on disk
        After deletion the person will no longer be recognized.
        """
        print(f"\n[DEL] Deleting: '{person_name}'")

        # ── Remove from in-memory store ───────────────────────────────────
        with self.lock:
            before_count = len(self.registered_faces)
            self.registered_faces = [
                f for f in self.registered_faces
                if f['name'].lower() != person_name.lower()
            ]
            removed_count = before_count - len(self.registered_faces)

        if removed_count == 0:
            return {
                "success": False,
                "removed_count": 0,
                "error": f"Person '{person_name}' not found in the dataset."
            }

        # ── Remove dataset folder ─────────────────────────────────────────
        # Try both the sanitized name and the raw name
        safe_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '_', '-')).strip()
        candidate_dirs = [
            self.dataset_root / safe_name,
            self.dataset_root / person_name,
        ]

        folder_deleted = False
        for dataset_dir in candidate_dirs:
            if dataset_dir.exists():
                try:
                    shutil.rmtree(dataset_dir, ignore_errors=True)
                    print(f"[DEL] Dataset folder removed: {dataset_dir}")
                    folder_deleted = True
                    break
                except Exception as e:
                    print(f"[WARN] Could not remove dataset folder '{dataset_dir}': {e}")

        if not folder_deleted:
            print(f"[WARN] Dataset folder for '{person_name}' not found or already removed.")

        result = {
            "success": True,
            "removed_count": removed_count,
            "person": person_name,
            "folder_deleted": folder_deleted
        }
        print(f"[OK] Deleted '{person_name}' ({removed_count} embeddings removed from memory)")
        return result

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def rescan_dataset(self) -> Dict[str, Any]:
        """
        Clear the in-memory embedding store and re-scan the dataset folder from disk.
        Useful when:
          - Images were manually added to a person's folder
          - Person folders were renamed
          - Server didn't pick up newly registered persons
        Returns dataset stats after reload.
        """
        print("\n[RESCAN] Clearing in-memory store and re-scanning dataset folder…")
        with self.lock:
            self.registered_faces = []
        self._scan_dataset()
        stats = self.get_database_stats()
        print(f"[RESCAN] Done: {stats['unique_persons']} persons, {stats['total_embeddings']} embeddings")
        return stats

    def get_registered_persons(self) -> List[Dict[str, Any]]:
        """
        Return a sorted list of all registered persons with metadata.
        Only includes persons whose dataset folder still exists on disk.
        """
        with self.lock:
            persons: Dict[str, Dict] = {}
            for item in self.registered_faces:
                n = item["name"]
                # Skip stale entries where the dataset folder was manually deleted
                folder = Path(item.get("dataset_path", ""))
                if folder and not folder.exists():
                    continue
                if n not in persons:
                    persons[n] = {
                        "name": n,
                        "gender": item.get("gender", "Unknown"),
                        "count": 0,
                        "dataset_path": item.get("dataset_path", "")
                    }
                persons[n]["count"] += 1
            return sorted(persons.values(), key=lambda x: x["name"])

    def get_database_stats(self) -> Dict[str, Any]:
        """Return recognition statistics."""
        persons = self.get_registered_persons()
        return {
            "total_embeddings": len(self.registered_faces),
            "unique_persons": len(persons),
            "dataset_root": str(self.dataset_root),
            "persons": {
                p["name"]: {"count": p["count"], "gender": p["gender"]}
                for p in persons
            }
        }

    def _verify_person_in_db(self, name: str) -> bool:
        """Check whether a person exists in the current in-memory store."""
        persons = self.get_registered_persons()
        return any(p['name'].lower() == name.lower() for p in persons)

    # ------------------------------------------------------------------
    # Legacy / dataset builder (kept for backwards compatibility)
    # ------------------------------------------------------------------
    def load_dataset(self, dataset_path: str) -> Dict[str, List[str]]:
        """Scan a folder tree and return {person_name: [image_paths]}."""
        dataset: Dict[str, List[str]] = {}
        dataset_path_obj = Path(dataset_path)

        if not dataset_path_obj.exists():
            print(f"[ERROR] Dataset path does not exist: {dataset_path}")
            return dataset

        for person_folder in dataset_path_obj.iterdir():
            if person_folder.is_dir():
                image_files = list(self._find_images(person_folder))
                if image_files:
                    dataset[person_folder.name] = [str(p) for p in image_files]

        return dataset

    def build_embeddings_database(self, dataset_path: str,
                                  output_path: Optional[str] = None) -> bool:
        """
        (Re-)build the recognition store from a dataset folder.
        Useful for bulk import / recovery.
        """
        print("\n[BUILD] Building embeddings from dataset…")
        dataset = self.load_dataset(dataset_path)
        if not dataset:
            print("[ERROR] No dataset found")
            return False

        total = sum(len(imgs) for imgs in dataset.values())
        done  = 0
        ok    = 0

        for person_name, image_paths in dataset.items():
            print(f"\n[BUILD] Processing {person_name} ({len(image_paths)} images)…")
            for image_path in image_paths:
                done += 1
                print(f"  [{done}/{total}] {os.path.basename(image_path)}…", end=" ")
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        print("[SKIP] cannot read")
                        continue
                    embedding = self._extract_embedding_from_stored_image(img)
                    if embedding is not None:
                        with self.lock:
                            self.registered_faces.append({
                                "name": person_name,
                                "embedding": embedding,
                                "source_image": image_path,
                                "dataset_path": str(Path(image_path).parent)
                            })
                        ok += 1
                        print("[OK]")
                    else:
                        print("[SKIP] no embedding")
                except Exception as e:
                    print(f"[ERROR] {e}")

        print(f"\n[BUILD] Done: {ok}/{total} embeddings added.")
        return ok > 0

    def reset_database(self) -> bool:
        """Delete ALL registered persons — clears in-memory data and dataset folder."""
        print("[RESET] Clearing all data…")

        with self.lock:
            self.registered_faces = []

        if self.dataset_root.exists():
            try:
                shutil.rmtree(self.dataset_root, ignore_errors=True)
                self.dataset_root.mkdir(parents=True, exist_ok=True)
                print(f"[RESET] Dataset folder cleared: {self.dataset_root}")
            except Exception as e:
                print(f"[WARN] Could not clear dataset folder: {e}")

        print("[RESET] Done.")
        return True
