# project_cnn_flutter

# 🧠 Project CNN + Flutter – Classification d'animaux

Ce projet montre comment entraîner un modèle **CNN** sur des images réelles
(d'éléphants, lions, etc.) et l'exporter en **TensorFlow Lite** pour l'utiliser
dans une application **Flutter**.

## 📂 Structure du dépôt

```text
Project_CNN_Flutter/
├─ donnees/
│   ├─ entrainement/   # images pour l'entraînement (par classe)
│   ├─ validation/     # 10–15 % des images d'entraînement
│   └─ test/           # images jamais vues pour l'évaluation finale
├─ src/
│   ├─ create_validation_split.py  # crée l'ensemble de validation
│   ├─ train_cnn.py                # charge le dataset et entraîne la CNN
│   └─ evaluate_cnn.py (optionnel) # analyses avancées
├─ models/
│   ├─ cnn_animaux.h5
│   ├─ cnn_animaux.tflite
│   └─ labels.txt
└─ README.md

## 🚀 Mises à jour récentes (Mars 2026)

Cette application a fait l'objet d'une révision majeure pour corriger des bugs critiques de classification et améliorer l'expérience utilisateur avec l'Intelligence Artificielle.

### 🐛 Corrections de Bugs (Bug Fixes)
- **Correction du "Biais de l'Éléphant" (Rotation EXIF)** : Résolution d'un bug critique où les photos de l'appareil photo étaient systématiquement classées comme "Éléphant". L'application applique désormais `img.bakeOrientation()` pour corriger la rotation EXIF des smartphones modernes avant d'effectuer le recadrage (crop), garantissant que le réseau de neurones regarde l'animal et non le mur.
- **Normalisation TFLite (EfficientNetB3)** : Ajustement des valeurs RGB pour qu'elles correspondent au format brut `[0, 255]` attendu par l'architecture EfficientNet, évitant ainsi l'explosion des tenseurs causée par le format MobileNet `[-1, 1]`.

### ✨ Nouvelles Fonctionnalités (Features)
- **Filtre Anti-Humains (Google ML Kit)** : Intégration de `google_mlkit_face_detection` pour scanner les images _avant_ l'inférence. Si un visage humain est détecté (ex: un selfie), l'application bloque l'IA animale et affiche un message humoristique, résolvant élégamment le problème du "Closed-Set" du modèle CNN.
- **Heuristiques Anti-Dessins (False-Positive filtering)** : Ajout d'algorithmes vérifiant le ratio de pixels blancs (`whiteRatio`) et la saturation moyenne (`avgSaturation`). L'application rejette désormais intelligemment les caricatures, dessins animés ou fonds d'écran trop saturés, grâce à une exigence de confiance stricte (>98%).
- **Interface Utilisateur Bilingue & Audio** : Prise en charge de l'espagnol (ES) et du français (FR). Ajout d'effets sonores distincts pour les prédictions réussies (+98% de confiance) et pour les erreurs/rejets.
