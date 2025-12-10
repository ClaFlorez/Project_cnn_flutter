# Classification d’animaux avec EfficientNetB3 (Flutter-ready)

Ce projet implémente un modèle de classification d’images pour 6 animaux sauvages,
entraîné avec **EfficientNetB3** et exporté en **TensorFlow Lite** pour une utilisation
dans une application mobile Flutter.

## 🎯 Objectif

- Classifier automatiquement des images en 6 classes :
  - `elephant`
  - `girafe`
  - `leopard`
  - `rhino`
  - `tigre`
  - `zebre`
- Atteindre une précision **≥ 99 %** sur le jeu de test.
- Déployer le modèle sur mobile avec **Flutter + TFLite**.

---

## 📂 Organisation du projet

- `notebooks/`
  - `cnn_animals_efficientnetb3.ipynb`  
    Notebook complet d’entraînement (Phase 1 + Phase 2, visualisation, export TFLite).
- `models/`
  - `cnn_model_animals_2025.keras` – modèle Keras complet.
  - `cnn_model_animals_2025_best.keras` – meilleurs poids (EarlyStopping).
  - `cnn_model_animals_2025.tflite` – modèle optimisé pour mobile.
  - `model_labels.txt` – liste des classes, une par ligne.
- `flutter_app/`
  - Exemple d’intégration via `tflite_flutter`.

---

## 🧠 Architecture du modèle

### 1. Backbone

- **EfficientNetB3** pré-entraîné sur ImageNet
- Couches de base gelées dans la Phase 1
- Fine-tuning partiel dans la Phase 2

### 2. Tête de classification

```text
GlobalAveragePooling2D
Dropout(0.3)
Dense(256) + BatchNormalization + ReLU
Dropout(0.5)
Dense(6, activation="softmax")
