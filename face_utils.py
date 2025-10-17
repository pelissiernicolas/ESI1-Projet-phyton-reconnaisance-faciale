# face_utils.py (OpenCV-only)
import os, json, cv2, numpy as np
from typing import Optional, Tuple, List
from PIL import Image

ENCODINGS_PATH = os.path.join("data","encodings","lbph_model.yml")
META_PATH      = os.path.join("data","encodings","meta.json")
IMAGES_DIR     = os.path.join("data","images")
CASCADE_PATH   = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

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
    """Ajoute (ou remplace) une image d'étudiant puis (ré)entraîne le modèle."""
    _ensure_dirs()
    # On suppose que l'image est déjà au bon emplacement (copiée par l'app)
    # Ici on (ré)entraîne simplement.
    train_or_update_model()

def match_face(probe_image_path: str, threshold: float = 60.0) -> Tuple[Optional[str], float]:
    """Retourne (best_image_fullpath, score) si reconnu, sinon (None, score). Score LBPH plus bas = mieux."""
    if not (os.path.exists(ENCODINGS_PATH) and os.path.exists(META_PATH)):
        # pas de modèle → tenter un entraînement rapide
        ok = train_or_update_model()
        if not ok:
            return None, 1e9

    face = _detect_face_gray(_read_pil(probe_image_path))
    if face is None:
        return None, 1e9

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(ENCODINGS_PATH)
    label, confidence = rec.predict(face)  # confidence: plus petit = meilleur
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

