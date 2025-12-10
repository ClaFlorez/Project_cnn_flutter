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

3. Entrée / Sortie

Entrée : image RGB 224x224x3 en float32, valeurs 0–255.

Sortie : vecteur de 6 probabilités (softmax).

📊 Jeu de données

6 classes : elephant, girafe, leopard, rhino, tigre, zebre

Environ :

20 400 images pour l’entraînement

3 600 images pour la validation

6 000 images pour le test (1 000 par classe)

Distribution équilibrée entre les classes.

Les images sont organisées par répertoires :

data/
  train/
    elephant/
    girafe/
    leopard/
    rhino/
    tigre/
    zebre/
  validation/
    ...
  test/
    ...

⚙️ Entraînement du modèle
Phase 1 – Entraînement de la tête (backbone gelé)

Backbone EfficientNetB3 gelé (non entraînable).

Seules les couches de la tête sont entraînées.

Optimiseur : Adam(learning_rate=1e-3)

Loss : categorical_crossentropy

Métriques : accuracy, precision, recall

Data augmentation :

rotation_range=15

width_shift_range=0.10

height_shift_range=0.10

zoom_range=0.15

shear_range=0.10

horizontal_flip=True

brightness_range=[0.85, 1.15]

Résultats Phase 1 :

Accuracy validation ≈ 98.5 % dès les premiers epochs.

Phase 2 – Fine-tuning du backbone

On déverrouille ~80 % des couches du backbone EfficientNetB3.

Seules les ~20 % premières couches restent gelées.

Nouveau learning rate : 5e-5 → 1e-4 (ReduceLROnPlateau).

Callbacks :

EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)

ModelCheckpoint(..._best.keras, save_best_only=True)

ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3).

Résultat final (meilleur epoch ≈ 25/50) :

Accuracy train ≈ 99.9 %

Accuracy validation ≈ 99.5 %

✅ Résultats sur le jeu de test
Classification report
Classe	Precision	Recall	F1-score	Support
elephant	0.99	0.98	0.98	1000
girafe	1.00	0.99	1.00	1000
leopard	0.99	0.99	0.99	1000
rhino	0.98	0.99	0.98	1000
tigre	1.00	1.00	1.00	1000
zebre	0.99	1.00	1.00	1000

Accuracy globale : ~99 %

Macro avg F1 : 0.99

Weighted avg F1 : 0.99

Matrice de confusion (résumé)

Rhino vs Elephant : très peu de confusions (≤ 14/1000).

Tigre et Zebre quasiment parfaits (997/1000 correctement classés).

Girafe et Leopard extrêmement stables.

📱 Exportation TFLite

Le modèle final est exporté en :

# Exemple de code (dans le notebook)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("cnn_model_animals_2025.tflite", "wb") as f:
    f.write(tflite_model)


La conversion a été vérifiée en comparant les prédictions Keras vs TFLite :

Différence absolue maximale ≈ 0.000119

Les sorties TFLite sont pratiquement identiques à celles du modèle Keras.

📲 Intégration Flutter

Un guide détaillé est disponible dans instructions_flutter.txt.
Résumé :

Plugin : tflite_flutter

Input Tensor :

Shape : [1, 224, 224, 3]

Type : float32

Pixels : 0–255 (pas de division par 255)

Output Tensor :

Shape : [1, 6]

Probabilités softmax pour chaque classe

Labels : chargés depuis model_labels.txt

🔬 Comparaison MobileNetV2 vs EfficientNetB3

Une première version du projet utilisait MobileNetV2 :

Accuracy test ≈ 97–98 %

Confusion plus importante entre elephant et rhino.

La migration vers EfficientNetB3 + fine-tuning :

A augmenté la précision globale à ~99 %

A fortement réduit les confusions entre classes proches.

Offre un meilleur compromis précision / robustesse pour déploiement mobile.

🚀 Reproduire l’expérience

Cloner le dépôt.

Placer le dataset dans data/train, data/validation, data/test.

Ouvrir le notebook dans Google Colab.

Lancer les sections dans l’ordre :

Préparation des données

Construction du modèle EfficientNetB3

Phase 1 (tête)

Phase 2 (fine-tuning)

Évaluation + visualisation

Export TFLite

Tester le modèle dans Flutter avec cnn_model_animals_2025.tflite.

✨ Remerciements

Projet conçu et entraîné par Claudia (Claud-IA),
avec objectif pédagogique (IA + mobile) et déploiement dans une application Flutter
de reconnaissance d’animaux.
