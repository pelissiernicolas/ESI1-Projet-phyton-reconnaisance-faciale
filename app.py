# app.py
import os, csv, time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2

# IMPORTANT : utiliser la version OpenCV-only de face_utils.py
# (qui expose IMAGES_DIR, add_or_update_image, match_face)
from face_utils import IMAGES_DIR, add_or_update_image, match_face, detect_and_match_bgr
from face_utils import IMAGES_DIR, add_or_update_image, match_face, detect_and_match_bgr


STUDENTS_CSV = os.path.join("data", "students.csv")
TMP_DIR = os.path.join("data", "tmp")
ADMIN_USER = "admin"
ADMIN_PASS = "admin"

def load_students():
    """
    Charge students.csv en détectant ; ou tab, nettoie l'en-tête et
    ignore les lignes sans id. Retourne une liste de dicts propres.
    """
    rows = []
    if not os.path.exists(STUDENTS_CSV):
        return rows

    # Sniff séparateur ; ou tab
    with open(STUDENTS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        sep = "\t" if ("\t" in first and ";" not in first) else ";"
        f.seek(0)
        reader = csv.reader(f, delimiter=sep)
        data = list(reader)

    if not data:
        return rows

    header = [h.strip().lstrip("\ufeff").lower() for h in data[0]]

    def idx(name, alts=()):
        for cand in (name, *alts):
            if cand in header:
                return header.index(cand)
        return -1

    i_id     = idx("id", ("ï»¿id",))
    i_prenom = idx("prenom", ("prénom",))
    i_nom    = idx("nom")
    i_solde  = idx("solde")

    for line in data[1:]:
        if not any(line):
            continue
        def safe(i): return line[i].strip() if 0 <= i < len(line) else ""
        rid = safe(i_id)
        if not rid:           # saute les lignes sans id
            continue
        rows.append({
            "id": rid,
            "prenom": safe(i_prenom),
            "nom": safe(i_nom),
            "solde": safe(i_solde) or "0"
        })
    return rows

def save_students(rows):
    """Sauvegarde students.csv (séparateur ';', encodage UTF-8 BOM)."""
    os.makedirs(os.path.dirname(STUDENTS_CSV), exist_ok=True)
    with open(STUDENTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","prenom","nom","solde"], delimiter=";")
        w.writeheader()
        for row in rows:
            w.writerow(row)

def ensure_dirs():
    """Crée les répertoires data/images si besoin."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STUDENTS_CSV), exist_ok=True)


class AdminCamFrame(ttk.Frame):
    """
    Onglet admin : flux live avec détection visages + score LBPH + identité.
    Boutons Démarrer / Arrêter. Le flux s'exécute avec after() (tkinter-safe).
    """
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.cap = None
        self.running = False
        self.label = ttk.Label(self, text="Flux arrêté.")
        self.label.pack(pady=6)

        btns = ttk.Frame(self)
        btns.pack(pady=6)
        ttk.Button(btns, text="Démarrer caméra", command=self.start).pack(side="left", padx=4)
        ttk.Button(btns, text="Arrêter", command=self.stop).pack(side="left", padx=4)

        self.video = ttk.Label(self)  # contiendra l'image
        self.video.pack(pady=8)
        self._imtk = None

        # paramètres d'affichage
        self.threshold = 60.0  # plus grand = plus permissif

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam", "Impossible d'ouvrir la caméra.")
            self.cap = None
            return
        self.running = True
        self.label.configure(text="Caméra en cours...")
        self.after(1, self._loop)

    def stop(self):
        self.running = False
        self.label.configure(text="Flux arrêté.")
        if self.cap:
            self.cap.release()
            self.cap = None

    def _loop(self):
        if not self.running or not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.label.configure(text="Lecture caméra échouée.")
            self.stop()
            return

        # Détection + matching (OpenCV-only)
        results = detect_and_match_bgr(frame, threshold=self.threshold)

        # Dessin des boxes + infos
        for r in results:
            (x, y, w, h) = r["bbox"]
            color = (0, 180, 0) if r["ok"] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            caption = f"score={r['score']:.1f}"
            if r["ok"]:
                who = f"{r.get('prenom') or ''} {r.get('nom') or ''}".strip()
                sid = r.get("id") or "?"
                caption = f"{who} (ID {sid})  {caption}"
            cv2.putText(frame, caption, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Convertir BGR -> RGB -> PhotoImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((800, int(800 * img.height / img.width))) if img.width > 800 else img
        self._imtk = ImageTk.PhotoImage(img)
        self.video.configure(image=self._imtk)

        # next frame
        self.after(33, self._loop)  # ~30 fps

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DSN - Contrôle d'accès restaurant scolaire")
        self.geometry("860x640")
        self.minsize(800, 560)
        ensure_dirs()

        # --- Barre login (optionnel) ---
        self.login_frame = ttk.Frame(self, padding=10)
        self.login_frame.pack(fill="x")
        ttk.Label(self.login_frame, text="Admin (optionnel) :").grid(row=0, column=0, sticky="w")
        self.e_user = ttk.Entry(self.login_frame, width=14)
        self.e_user.grid(row=0, column=1, padx=4)
        self.e_pass = ttk.Entry(self.login_frame, show="*", width=14)
        self.e_pass.grid(row=0, column=2, padx=4)
        ttk.Button(self.login_frame, text="Connexion", command=self.do_login).grid(row=0, column=3, padx=4)

        # --- Onglets ---
        self.tabs = ttk.Notebook(self)
        self.tab_enroll = ttk.Frame(self.tabs)
        self.tab_access = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_enroll, text="Ajout étudiant")
        self.tabs.add(self.tab_access, text="Contrôle d'accès")
        self.tabs.pack(fill="both", expand=True, padx=8, pady=8)

        # Prépare l'onglet admin (sans l'ajouter tant que pas loggé)
        self.tab_admin = AdminCamFrame(self)
        self._admin_added = False

        # ========= Onglet Ajout =========
        self.var_id = tk.StringVar()
        self.var_first = tk.StringVar()
        self.var_last = tk.StringVar()
        self.var_balance = tk.StringVar(value="10.00")
        self.selected_image_path = None
        self._imtk_preview = None

        frm = ttk.Frame(self.tab_enroll, padding=10)
        frm.pack(fill="x", anchor="w")

        ttk.Label(frm, text="ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_id, width=12).grid(row=0, column=1, padx=5)

        ttk.Label(frm, text="Prénom").grid(row=0, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.var_first, width=18).grid(row=0, column=3, padx=5)

        ttk.Label(frm, text="Nom").grid(row=0, column=4, sticky="w")
        ttk.Entry(frm, textvariable=self.var_last, width=18).grid(row=0, column=5, padx=5)

        ttk.Label(frm, text="Solde (€)").grid(row=0, column=6, sticky="w")
        ttk.Entry(frm, textvariable=self.var_balance, width=10).grid(row=0, column=7, padx=5)

        btns = ttk.Frame(self.tab_enroll, padding=10)
        btns.pack(fill="x", anchor="w")
        ttk.Button(btns, text="Importer image…", command=self.pick_image).pack(side="left", padx=4)
        ttk.Button(btns, text="Capture webcam", command=self.capture_webcam).pack(side="left", padx=4)
        ttk.Button(btns, text="Capturer série", command=lambda: self.capture_serie()).pack(side="left", padx=4)
        ttk.Button(btns, text="Enregistrer étudiant", command=self.save_student).pack(side="left", padx=4)

        self.preview_label = ttk.Label(self.tab_enroll)
        self.preview_label.pack(pady=10)

        # ========= Onglet Contrôle =========
        ttk.Label(self.tab_access, text="Contrôle d'accès : utilisez la webcam ou importez une image.").pack(pady=8)

        access_btns = ttk.Frame(self.tab_access, padding=10)
        access_btns.pack(fill="x", anchor="w")
        ttk.Button(access_btns, text="Scanner webcam", command=self.scan_webcam).pack(side="left", padx=4)
        ttk.Button(access_btns, text="Importer image…", command=self.check_image_file).pack(side="left", padx=4)

        self.access_result = ttk.Label(self.tab_access, font=("Segoe UI", 12, "bold"))
        self.access_result.pack(pady=12)

    # ---------- Login ----------
    def do_login(self):
        u = self.e_user.get().strip()
        p = self.e_pass.get().strip()
        if u == ADMIN_USER and p == ADMIN_PASS:
            messagebox.showinfo("Connexion", "Accès administrateur OK")
            # Ajoute l'onglet Admin si pas déjà présent
            if not getattr(self, '_admin_added', False):
                self.tabs.add(self.tab_admin, text="Admin caméra")
                self._admin_added = True
                self.tabs.select(self.tab_admin)
        else:
            messagebox.showwarning("Connexion", "Identifiants incorrects (mode admin facultatif).")

    # ---------- Ajout étudiant ----------
    def pick_image(self):
        p = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png")]
        )
        if not p:
            return
        self.selected_image_path = p
        self.show_preview(p)

    def capture_webcam(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW pour Windows
        if not cap.isOpened():
            messagebox.showerror("Webcam", "Impossible d'ouvrir la webcam.")
            return
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Webcam", "Capture échouée.")
            return
        tmp = os.path.join(TMP_DIR, f"capture_{int(time.time())}.jpg")
        cv2.imwrite(tmp, frame)
        self.selected_image_path = tmp
        self.show_preview(tmp)

    def capture_serie(self, n=8, interval_ms=300):
        """Capture une série d'images (n) à interval_ms millisecondes et les enregistre
        sous data/images/<id>_<prenom>_<nom>_001.jpg ... _00n.jpg puis réentraîne.
        """
        sid = self.var_id.get().strip()
        first = self.var_first.get().strip()
        last = self.var_last.get().strip()
        if not (sid and first and last):
            messagebox.showwarning("Champs", "Remplissez ID / Prénom / Nom avant de capturer la série.")
            return

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Webcam", "Impossible d'ouvrir la webcam.")
            return
        images = []
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            images.append(frame.copy())
            # petit délai (bloquant léger mais permet captures régulières)
            time.sleep(interval_ms / 1000.0)
            # garde l'UI réactive
            try:
                self.update_idletasks()
                self.update()
            except Exception:
                pass
        cap.release()

        if not images:
            messagebox.showerror("Capture série", "Aucune image capturée.")
            return

        os.makedirs(IMAGES_DIR, exist_ok=True)
        last_dest = None
        for i, fr in enumerate(images, 1):
            dest = os.path.join(IMAGES_DIR, f"{sid}_{first}_{last}_{i:03}.jpg")
            try:
                cv2.imwrite(dest, fr)
                last_dest = dest
            except Exception as e:
                # on continue malgré les erreurs d'écriture
                print(f"Erreur écriture image série: {e}")

        if last_dest:
            try:
                add_or_update_image(last_dest)
            except Exception as e:
                messagebox.showwarning("Modèle", f"Série enregistrée, mais réentraînement incomplet : {e}")
        messagebox.showinfo("Capture série", f"{len(images)} images enregistrées pour l'étudiant {first} {last}.")

    def show_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((520, 390))
            self._imtk_preview = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self._imtk_preview)
        except Exception as e:
            messagebox.showerror("Image", f"Impossible d'afficher l'aperçu : {e}")

    def save_student(self):
        sid   = self.var_id.get().strip()
        first = self.var_first.get().strip()
        last  = self.var_last.get().strip()
        bal   = self.var_balance.get().strip()

        if not (sid and first and last and bal and self.selected_image_path):
            messagebox.showwarning("Champs", "Remplissez tous les champs et choisissez une image.")
            return

        # Valider le solde
        try:
            float(bal)
        except ValueError:
            messagebox.showwarning("Solde", "Le solde doit être un nombre (ex: 10.00).")
            return

        # Enregistrer l'image dans data/images avec convention <id>_<prenom>_<nom>.jpg
        dest_name = f"{sid}_{first}_{last}.jpg"
        dest_path = os.path.join(IMAGES_DIR, dest_name)
        try:
            Image.open(self.selected_image_path).convert("RGB").save(dest_path)
        except Exception as e:
            messagebox.showerror("Image", f"Impossible d'enregistrer l'image : {e}")
            return

        # Mise à jour students.csv
        rows = load_students()
        updated = False
        for r in rows:
            if r["id"] == sid:
                r["prenom"], r["nom"], r["solde"] = first, last, bal
                updated = True
                break
        if not updated:
            rows.append({"id": sid, "prenom": first, "nom": last, "solde": bal})
        save_students(rows)

        # (Ré)entraîner le modèle LBPH via face_utils
        try:
            add_or_update_image(dest_path)
        except Exception as e:
            messagebox.showwarning("Modèle", f"Enregistré, mais réentraînement incomplet : {e}")

        messagebox.showinfo("Ajout", f"Étudiant {first} {last} (ID {sid}) enregistré.")

    # ---------- Contrôle d'accès ----------
    def scan_webcam(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("Webcam", "Impossible d'ouvrir la webcam.")
            return
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("Webcam", "Capture échouée.")
            return
        tmp = os.path.join(TMP_DIR, f"probe_{int(time.time())}.jpg")
        cv2.imwrite(tmp, frame)
        self._check_probe(tmp)

    def check_image_file(self):
        p = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png")]
        )
        if not p:
            return
        self._check_probe(p)

    def _check_probe(self, probe_path):
        """
        Essaie d'identifier le visage.
        Si reconnu et solde >= 1.00 -> Autorisé + décrément du solde.
        Sinon -> Refusé (ou messages d'avertissement).
        """
        try:
            best, score = match_face(probe_path, threshold=70.0)  # LBPH: plus petit = mieux
        except Exception as e:
            self.access_result.configure(text=f"Erreur reconnaissance : {e}", foreground="orange")
            return

        if best is None:
            self.access_result.configure(text="Refusé : visage non reconnu.", foreground="red")
            return

        # best = chemin data/images/<id>_<prenom>_<nom>.jpg
        base = os.path.basename(best)
        try:
            sid, first, last = os.path.splitext(base)[0].split("_", 2)
        except Exception:
            self.access_result.configure(text="Reconnu mais fiche invalide (nom de fichier).", foreground="orange")
            return

        rows = load_students()
        for r in rows:
            rid = (r.get("id") or r.get("ï»¿id") or r.get("\ufeffid") or "").strip()
            if rid == sid:
                try:
                    solde = float(r.get("solde", "0"))
                except ValueError:
                    self.access_result.configure(text=f"Fiche corrompue : solde invalide pour {first} {last}", foreground="orange")
                    return

                if solde >= 1.0:
                    solde -= 1.0
                    r["solde"] = f"{solde:.2f}"
                    save_students(rows)
                    self.access_result.configure(
                        text=f"Autorisé : {first} {last} (ID {sid}) – Nouveau solde : {solde:.2f} €",
                        foreground="green"
                    )
                else:
                    self.access_result.configure(
                        text=f"Refusé (solde insuffisant) : {first} {last} – Solde : {solde:.2f} €",
                        foreground="red"
                    )
                return

        self.access_result.configure(text="Reconnu mais non trouvé en base.", foreground="orange")

if __name__ == "__main__":
    App().mainloop()
