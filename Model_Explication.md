
# 📖 Explication pas à pas du notebook `Model.ipynb`

Ce document explique **étape par étape** le code utilisé pour entraîner un modèle de classification d’images (animaux) avec TensorFlow/Keras dans Google Colab, puis l’exporter pour une application Flutter via TensorFlow Lite.

Il est conçu pour être ajouté à votre dépôt GitHub, par exemple sous le nom :  
`docs/Model_Explication.md` ou `Model_Explication.md`.

---

## 1. Configuration de TensorFlow et du GPU

La première section du notebook vérifie la version de TensorFlow et la disponibilité du GPU, puis configure l’allocation mémoire pour éviter des erreurs d’out of memory :

```python
import tensorflow as tf

print("Version de TensorFlow:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU configuré avec memory growth.")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ Aucun GPU détecté, entraînement en CPU.")
```

**But :**
- Vérifier que Colab utilise bien un GPU.
- Configurer le “memory growth” pour que TensorFlow n’alloue pas toute la mémoire GPU d’un coup.

---

## 2. Imports principaux et paramètres globaux

On importe toutes les bibliothèques nécessaires : Numpy, Matplotlib, Seaborn, métriques de scikit-learn, ainsi que les modules Keras/TensorFlow.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import datetime
from pathlib import Path
import os
```

Ensuite, on définit un dictionnaire de paramètres (taille d’image, batch size, nombre de classes, etc.) :

```python
PARAMS = {
    "img_height": 128,
    "img_width": 128,
    "img_channels": 3,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_classes": 6,   # elephant, girafe, leopard, rhino, tigre, zebre
    "epochs": 20
}
```

**But :**
- Centraliser tous les hyperparamètres dans une seule structure.
- Permettre de modifier facilement la taille d’image, le batch size ou le learning rate.

---

## 3. Récupération du dataset depuis le dépôt GitHub

Le notebook clone le dépôt GitHub pour récupérer les images déjà organisées en 3 dossiers : `entrainement`, `validation`, `test`.

```python
!rm -rf /content/Project_cnn_flutter
!git clone https://github.com/ClaFlorez/Project_cnn_flutter.git /content/Project_cnn_flutter

BASE_DIR = "/content/Project_cnn_flutter"

DATA_PATHS = {
    "train":      f"{BASE_DIR}/entrainement",
    "validation": f"{BASE_DIR}/validation",
    "test":       f"{BASE_DIR}/test",
}
```

On vérifie ensuite que les dossiers existent et on affiche leur contenu :

```python
from pathlib import Path

print("\n📂 Vérification des dossiers du dataset :")
for split, path in DATA_PATHS.items():
    p = Path(path)
    print(f"- {split}: {p} → existe:", p.exists())
    if p.exists():
        print("  Contenu:", os.listdir(p))
```

**But :**
- S’assurer que la structure du dataset est correcte après le clonage.
- Détecter les classes en se basant sur les sous-dossiers de `entrainement`.

---

## 4. Générateurs de données et augmentation

On utilise `ImageDataGenerator` pour :
- appliquer une **normalisation** systématique,
- et une **augmentation de données** sur l’ensemble d’entraînement.

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)
```

Création des générateurs :

```python
train_generator = train_datagen.flow_from_directory(
    DATA_PATHS["train"],
    target_size=(PARAMS["img_height"], PARAMS["img_width"]),
    batch_size=PARAMS["batch_size"],
    class_mode="categorical",
    shuffle=True
)

validation_generator = val_test_datagen.flow_from_directory(
    DATA_PATHS["validation"],
    target_size=(PARAMS["img_height"], PARAMS["img_width"]),
    batch_size=PARAMS["batch_size"],
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    DATA_PATHS["test"],
    target_size=(PARAMS["img_height"], PARAMS["img_width"]),
    batch_size=PARAMS["batch_size"],
    class_mode="categorical",
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print("Classes détectées :", class_names)
```

**But :**
- Normaliser les pixels dans [0,1].
- Enrichir le dataset d’apprentissage via des transformations aléatoires.
- Charger automatiquement les étiquettes en fonction des noms de dossiers.

---

## 5. Définition du modèle CNN

Le cœur du notebook est la fonction `create_cnn_model`, qui construit un modèle CNN compatible Keras 3 :

```python
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Bloc 1
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    # Bloc 2
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    # Bloc 3
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    # Couches denses
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Couche de sortie
    model.add(Dense(num_classes, activation='softmax'))

    # Compilation
    optimizer = Adam(learning_rate=PARAMS["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

Création d’une instance du modèle :

```python
input_shape = (PARAMS["img_height"], PARAMS["img_width"], PARAMS["img_channels"])
model = create_cnn_model(input_shape, PARAMS["num_classes"])
model.summary()
```

**But :**
- Construire une architecture CNN adaptée à des images 128×128×3.
- Utiliser BatchNorm + Dropout pour stabiliser et régulariser l’apprentissage.
- Utiliser Adam avec un learning rate de 0.001 (valeur standard).

---

## 6. Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)

Pour fiabiliser l’entraînement, le notebook configure plusieurs callbacks :

```python
import datetime

# Nom de base pour les fichiers (timestamp)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"cnn_model_{timestamp}"

SAVE_PATH = "/content/drive/MyDrive/models/"

checkpoint_path = SAVE_PATH + model_name + "_best.h5"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]
```

**But :**
- Sauvegarder automatiquement le **meilleur modèle**.
- Arrêter l’entraînement lorsque la validation ne s’améliore plus.
- Diminuer le learning rate en cas de plateau.

---

## 7. Entraînement du modèle

```python
history = model.fit(
    train_generator,
    epochs=PARAMS["epochs"],
    validation_data=validation_generator,
    callbacks=callbacks
)
```

**But :**
- Lancer l’apprentissage sur les images d’animaux.
- Suivre la loss et l’accuracy sur train et validation à chaque époque.

---

## 8. Évaluation, courbes et matrice de confusion

### 8.1 Évaluation sur le set de test

```python
test_loss, test_acc = model.evaluate(test_generator)
print(f"Accuracy sur le set de test: {test_acc:.4f}")
```

### 8.2 Rapport de classification et matrice de confusion

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

print(classification_report(y_true, y_pred, target_names=class_names))
```

```python
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.title("Matrice de confusion")
plt.tight_layout()
plt.savefig(SAVE_PATH + model_name + "_confusion_matrix.png")
plt.show()
```

### 8.3 Courbes d’apprentissage

```python
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(SAVE_PATH + model_name + "_training_history.png")
plt.show()
```

---

## 9. Sauvegarde du modèle et export en TFLite

```python
full_model_path = SAVE_PATH + model_name + "_full_model.h5"
model.save(full_model_path)
print("Modèle complet sauvegardé :", full_model_path)
```

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = SAVE_PATH + model_name + ".tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("Modèle TFLite sauvegardé :", tflite_path)
```

```python
labels_path = SAVE_PATH + model_name + "_labels.txt"
with open(labels_path, "w") as f:
    for label in class_names:
        f.write(label + "\n")

print("Labels sauvegardés dans :", labels_path)
```

---

## 10. Instructions Flutter générées automatiquement

```python
flutter_instructions_path = SAVE_PATH + model_name + "_flutter_instructions.txt"

instructions = f"""INTEGRATION FLUTTER DU MODELE: {model_name}.tflite

1. Copier les fichiers suivants dans votre projet Flutter:
   - assets/models/{model_name}.tflite
   - assets/models/{model_name}_labels.txt

2. Mettre à jour pubspec.yaml:

flutter:
  assets:
    - assets/models/{model_name}.tflite
    - assets/models/{model_name}_labels.txt

3. Utiliser tflite_flutter (ou équivalent) pour charger le modèle et faire des prédictions.
"""

with open(flutter_instructions_path, "w") as f:
    f.write(instructions)

print("Instructions Flutter sauvegardées dans :", flutter_instructions_path)
```

---

## 11. Résumé final

```python
summary = f"""FICHIERS GÉNÉRÉS:
-----------------
1. {model_name}_full_model.h5
2. {model_name}_best.h5
3. {model_name}.tflite
4. {model_name}_labels.txt
5. {model_name}_training_history.png
6. {model_name}_confusion_matrix.png
7. {model_name}_flutter_instructions.txt

PARAMÈTRES PRINCIPAUX:
----------------------
- Taille images: {PARAMS['img_height']}x{PARAMS['img_width']}
- Batch size: {PARAMS['batch_size']}
- Learning rate: {PARAMS['learning_rate']}
- Nombre classes: {PARAMS['num_classes']}
- Classes: {class_names}
- GPU utilisé: {len(gpus) > 0}

Tous les fichiers sont sauvegardés dans: {SAVE_PATH}
"""

print(summary)

with open(SAVE_PATH + model_name + "_summary.txt", "w") as f:
    f.write(summary)

print("\n✓ TP TERMINÉ AVEC SUCCÈS! ✓\n")
```

---

## 12. Intégration dans GitHub

Ajoutez ce fichier dans votre dépôt (par exemple `Model_Explication.md`) et créez un lien depuis votre `README.md` :

```markdown
Pour une explication détaillée du notebook d’entraînement, voir :
[Explication pas à pas du modèle](Model_Explication.md)
```

Ce document est pensé comme une **documentation pédagogique** de votre code, utile pour :
- les correcteurs du TP
- vos futurs projets
- toute personne souhaitant comprendre ou réutiliser votre pipeline.
