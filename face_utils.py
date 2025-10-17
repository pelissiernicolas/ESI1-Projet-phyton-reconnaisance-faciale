# face_utils.py (OpenCV-only)
import os, json, cv2, numpy as np
from typing import Optional, Tuple, List
from PIL import Image

# Paths (legacy LBPH kept for compatibility)
ENCODINGS_PATH = os.path.join("data","encodings","lbph_model.yml")
META_PATH      = os.path.join("data","encodings","meta.json")
EMB_PATH       = os.path.join("data","encodings","embeddings.json")
IMAGES_DIR     = os.path.join("data","images")
CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Optional modern embedding backend
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    insightface = None
    FaceAnalysis = None

_app = None

def _get_app():
    global _app
    if insightface is None or FaceAnalysis is None:
        return None
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l")
        # ctx_id=0 for CPU; if GPU/other, user can adjust
        try:
            _app.prepare(ctx_id=0, det_size=(320,320))
        except Exception:
            # try without ctx_id
            _app.prepare(det_size=(320,320))
    return _app

def _load_db():
    if not os.path.exists(EMB_PATH): return {}
    return json.load(open(EMB_PATH, "r", encoding="utf-8"))

def _save_db(d):
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    json.dump(d, open(EMB_PATH, "w", encoding="utf-8"))

def _img_bgr(path):
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

def _embedding_from_path(path) -> Optional[List[float]]:
    app = _get_app()
    if app is None:
        return None
    img = _img_bgr(path)
    faces = app.get(img)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return f.normed_embedding.astype(float).tolist()

def rebuild_embeddings_from_gallery():
    db = {}
    for f in sorted(os.listdir(IMAGES_DIR)):
        fl = f.lower()
        if not fl.endswith((".jpg",".jpeg",".png")) or fl.startswith("tmp_"):
            continue
        path = os.path.join(IMAGES_DIR, f)
        emb = _embedding_from_path(path)
        if emb is not None:
            db[f] = emb
    _save_db(db)
    return len(db)

def _ensure_dirs():
    os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

def _detect_face_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(CASCADE_PATH).detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face = gray[y:y+h, x:x+w]
    return _preprocess_face(face)


def _preprocess_face(gray_roi, size=(200,200)):
    """Prétraitement du visage en niveaux de gris : rejet flou, CLAHE, redimensionnement."""
    # Rejet si flou (variance du Laplacien)
    try:
        var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    except Exception:
        return None
    if var < 80:  # seuil à ajuster (50-120 selon le besoin)
        return None
    # Correction de contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_eq = clahe.apply(gray_roi)
    try:
        return cv2.resize(face_eq, size)
    except Exception:
        return None

def _read_pil(path):
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

def _load_meta():
    if os.path.exists(META_PATH):
        return json.load(open(META_PATH, "r", encoding="utf-8"))
    return {"labels":{}, "next_label":1}  # { "labels": {"1001_Nicolas_Pelissier.jpg": 1}, ... }

def _save_meta(meta):
    json.dump(meta, open(META_PATH, "w", encoding="utf-8"))

def train_or_update_model():
    """(Re)construit le modèle LBPH à partir du dossier images/"""
    _ensure_dirs()
    meta = _load_meta()
    images, labels = [], []
    meta["labels"].clear()
    label_next = 1
    for f in os.listdir(IMAGES_DIR):
        f_low = f.lower()
        if not f_low.endswith((".jpg",".jpeg",".png")): continue
        if f_low.startswith("tmp_"):        # ignorer les temporaires
            continue
        path = os.path.join(IMAGES_DIR, f)
        face = _detect_face_gray(_read_pil(path))
        if face is None: continue
        meta["labels"][f] = label_next
        images.append(face)
        labels.append(label_next)
        label_next += 1
    if not images:
        # rien à entraîner
        _save_meta(meta)
        return False
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(images, np.array(labels))
    rec.write(ENCODINGS_PATH)
    _save_meta(meta)
    return True

def add_or_update_image(image_path: str):
    """Après ajout d'une image : si InsightFace est disponible, calcule/update l'embedding,
    sinon retombe sur le pipeline LBPH (ré-entrainement complet)."""
    _ensure_dirs()
    app = _get_app()
    if app is not None:
        db = _load_db()
        emb = _embedding_from_path(image_path)
        if emb is not None:
            db[os.path.basename(image_path)] = emb
            _save_db(db)
            return
    # fallback LBPH
    train_or_update_model()

def match_face(probe_image_path: str, threshold: float = 0.40) -> Tuple[Optional[str], float]:
    """Si InsightFace disponible : calcule embedding du probe et compare cosine
    avec la base d'embeddings. Retourne (best_path, similarity) si sim >= threshold.
    Sinon, retombe sur le match LBPH existant (compat)."""
    # If insightface backend available, try embedding matching
    app = _get_app()
    if app is not None:
        db = _load_db()
        if not db:
            if rebuild_embeddings_from_gallery() == 0:
                return None, 0.0

        probe = _embedding_from_path(probe_image_path)
        if probe is None:
            return None, 0.0
        p = np.array(probe, dtype=np.float32)

        best_name, best_sim = None, -1.0
        for name, emb in db.items():
            e = np.array(emb, dtype=np.float32)
            sim = float(np.dot(p, e))  # embeddings normalisés -> dot = cosine
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_sim >= threshold:
            return os.path.join(IMAGES_DIR, best_name), best_sim
        return None, best_sim

    # Fallback to LBPH
    if not (os.path.exists(ENCODINGS_PATH) and os.path.exists(META_PATH)):
        ok = train_or_update_model()
        if not ok:
            return None, 0.0

    face = _detect_face_gray(_read_pil(probe_image_path))
    if face is None:
        return None, 0.0

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(ENCODINGS_PATH)
    label, confidence = rec.predict(face)  # confidence: plus petit = meilleur
    # For compatibility with threshold semantics, convert LBPH -> consider ok if confidence <= threshold
    if confidence > threshold:
        return None, float(confidence)

    meta = _load_meta()
    inv = {v:k for k,v in meta["labels"].items()}
    filename = inv.get(label)
    if not filename:
        return None, float(confidence)
    return os.path.join(IMAGES_DIR, filename), float(confidence)

# ---- Admin live detection ----
def _load_lbph():
    """Charge le modèle LBPH si présent, sinon tente un entrainement à partir de data/images."""
    if not (os.path.exists(ENCODINGS_PATH) and os.path.exists(META_PATH)):
        ok = train_or_update_model()
        if not ok:
            return None, None
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(ENCODINGS_PATH)
    meta = _load_meta()
    inv_labels = {v: k for k, v in meta["labels"].items()}
    return rec, inv_labels

def detect_and_match_bgr(frame_bgr, threshold: float = 60.0):
    """
    Analyse un frame BGR OpenCV et retourne une liste de résultats :
    [{ "bbox": (x,y,w,h), "ok": bool, "score": float, "match_path": str|None, "id": str|None, "prenom": str|None, "nom": str|None }]
    Note : score LBPH -> plus petit = meilleur; on considère ok si score <= threshold.
    """
    _ensure_dirs()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(CASCADE_PATH).detectMultiScale(gray, 1.1, 4)
    rec, inv = _load_lbph()
    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = _preprocess_face(face)
        ok = False
        score = 1e9
        match_path = None
        sid = first = last = None

        if rec is not None and face is not None:
            label, confidence = rec.predict(face)
            score = float(confidence)
            if score <= threshold and label in inv:
                filename = inv[label]
                match_path = os.path.join(IMAGES_DIR, filename)
                stem = os.path.splitext(filename)[0]
                try:
                    sid, first, last = stem.split("_", 3)
                except Exception:
                    pass
                ok = True

        results.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "ok": ok,
            "score": score,
            "match_path": match_path,
            "id": sid,
            "prenom": first,
            "nom": last,
        })
    return results


def _embedding_from_bgr(img_bgr) -> Optional[List[float]]:
    """Compute embedding from a BGR image (numpy array) using insightface app."""
    app = _get_app()
    if app is None:
        return None
    faces = app.get(img_bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return f.normed_embedding.astype(float).tolist()


def match_face_bgr(img_bgr, threshold: float = 0.40) -> Tuple[Optional[str], float]:
    """Match a BGR ROI/image against embeddings DB if available, else LBPH on ROI.
    Returns (best_image_fullpath, score) or (None, score).
    For embeddings: score is cosine similarity (bigger = better). For LBPH: score is confidence (smaller = better).
    """
    # Try embeddings first
    app = _get_app()
    if app is not None:
        db = _load_db()
        if not db:
            if rebuild_embeddings_from_gallery() == 0:
                return None, 0.0

        probe = _embedding_from_bgr(img_bgr)
        if probe is None:
            return None, 0.0
        p = np.array(probe, dtype=np.float32)

        best_name, best_sim = None, -1.0
        for name, emb in db.items():
            e = np.array(emb, dtype=np.float32)
            sim = float(np.dot(p, e))
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_name is None:
            return None, best_sim
        if best_sim >= threshold:
            return os.path.join(IMAGES_DIR, best_name), best_sim
        return None, best_sim

    # Fallback LBPH on ROI
    rec, inv = _load_lbph()
    if rec is None:
        return None, 0.0

    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None, 0.0
    face = _preprocess_face(gray)
    if face is None:
        return None, 0.0
    label, confidence = rec.predict(face)
    if confidence > threshold:
        return None, float(confidence)
    filename = inv.get(label)
    if not filename:
        return None, float(confidence)
    return os.path.join(IMAGES_DIR, filename), float(confidence)

