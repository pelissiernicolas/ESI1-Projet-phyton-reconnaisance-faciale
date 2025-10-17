# DSN — Contrôle d'accès (reconnaissance faciale)

Projet simple de contrôle d'accès pour un restaurant scolaire. Interface Tkinter pour :
- inscrire des étudiants (ID / prénom / nom / photo),
- capturer des séries d'images pour enrichir la galerie,
- scanner via webcam et autoriser/décrémenter le solde.

Le projet embarque deux modes de reconnaissance :
- fallback LBPH (OpenCV) — fonctionne sans dépendances lourdes mais moins robuste ;
- backend moderne InsightFace (embeddings ONNX) si installé — plus robuste et rapide.

## Arborescence importante
- `app.py` — application principale (Tkinter).
- `face_utils.py` — utilitaires de reconnaissance (LBPH + support InsightFace/embeddings).
- `data/` — données générées et persistantes :
  - `data/images/` — galerie (nommage `<id>_<prenom>_<nom>_xxx.jpg`).
  - `data/tmp/` — captures temporaires (probes, captures série).
  - `data/encodings/` — modèles/embeddings (`embeddings.json` ou `lbph_model.yml`).
  - `data/students.csv` — fichier des élèves (sous versionnement).

> `.gitignore` est configuré pour ignorer les images générées et les encodages (sauf `data/students.csv`).

## Prérequis
- Python 3.10+ (ou 3.8+ selon ton environnement)
- Windows (les instructions ci-dessous utilisent PowerShell)
- (optionnel mais recommandé) `insightface` + `onnxruntime` pour embeddings modernes

## Installation rapide (PowerShell)
Ouvrir PowerShell depuis le dossier du projet et lancer :

```powershell
# créer et activer venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# installer dépendances listées (fichier fourni)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si tu veux le backend InsightFace (recommandé pour de meilleures performances) :

```powershell
python -m pip install insightface onnxruntime
```

Note : `insightface` télécharge des modèles la première fois (connexion internet requise). Si `insightface` n'est pas présent, l'application utilisera LBPH.

## Lancer l'application

```powershell
py .\app.py
# ou
python app.py
```

## Fonctionnalités utiles
- Onglet "Ajout étudiant" : remplir `ID`, `Prénom`, `Nom` + importer image ou utiliser `Capture webcam` / `Capturer série` (8 images) pour enrichir l'entraînement.
- `Capturer série` crée plusieurs images nommées `ID_Prenom_Nom_001.jpg` etc. puis (ré)entraîne le modèle.
- ONGLET `Contrôle d'accès` : scanner webcam ou importer une image pour vérifier l'identité et décrémenter le solde si autorisé.
- Onglet `Admin caméra` : apparaît après connexion admin (login par défaut `admin` / `admin`). Affiche le flux live et les scores (sim=0.83 si embeddings activés).

## Commandes utilitaires
- Reconstruire les embeddings (parcours `data/images/`) :

```powershell
python -c "import face_utils; print(face_utils.rebuild_embeddings_from_gallery())"
```

- Vérifier / recalculer l'embedding d'une image ajoutée :

```powershell
python -c "import face_utils; face_utils.add_or_update_image('data/images/123_John_Doe_001.jpg')"
```

## Seuils et calibration
- Pour les embeddings (InsightFace) : seuil conseillé pour la similarité cosine ≈ 0.35–0.45 (par défaut 0.40).
- Pour LBPH : seuil LBPH est différent (plus petit = meilleur). Les valeurs par défaut sont réglées dans `face_utils.py`.

Calibre les seuils en testant ta caméra et tes images (variations de lumière, lunettes, profils).

## Dépannage
- Problème d'installation de `dlib` (erreur CMake) : privilégier l'option InsightFace (ONNX) qui évite `dlib`/compilation.
- Si la webcam ne s'ouvre pas : vérifier qu'aucune autre application ne l'utilise.
- Si aucun visage n'est détecté : tester avec photos centrées et bonne luminosité. Ajuster `CASCADE` ou utiliser InsightFace.

## Tests & développement
- Le code contient des fonctions importables dans `face_utils.py` pour tester reconstruction d'index :
  - `rebuild_embeddings_from_gallery()`
  - `add_or_update_image(path)`
  - `match_face(path, threshold)`

## Contribuer
- Respecte le `.gitignore` et ne commite pas les images/embeddings générés.
- Pour ajouter des tests unitaires, privilégier `pytest` et ajouter des fixtures minimalistes pour images factices.

---

Si tu veux, je peux :
- ajouter un script CLI `scripts/rebuild_embeddings.py` et un bouton UI pour déclencher la reconstruction depuis l'app ;
- ajouter un slider dans l'onglet Admin pour régler dynamiquement le seuil embeddings ;
- écrire un petit guide de calibration (série de captures pour mesurer TPR/FPR).

Dis-moi quelle option tu veux que j'implémente ensuite.
