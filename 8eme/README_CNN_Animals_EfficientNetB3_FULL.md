# 🧠 Classification d’Animaux par CNN — Documentation Complète
## EfficientNetB3 · TensorFlow · Vision par Ordinateur · Flutter

---

## 1. Vue d’ensemble du projet

Ce document décrit **en profondeur** l’ensemble du pipeline de classification d’images :
du chargement du dataset jusqu’au déploiement mobile.

Pipeline global :

```
Images → Prétraitement → CNN (EfficientNetB3)
→ Entraînement → Évaluation → TFLite → Flutter
```

---

## 2. Représentation mathématique des images

Une image RGB est représentée comme un tenseur :

```
(hauteur, largeur, canaux) = (224, 224, 3)
```

Chaque pixel est un triplet (R, G, B).
Après normalisation :

```
pixel_normalisé = pixel / 255
```

---

## 3. Rôle exact du CNN

Un CNN apprend une fonction :

```
f(x) = y
```

où :
- x : image d’entrée
- y : vecteur de probabilités

---

## 4. Convolution : fonctionnement mathématique

### 4.1 Définition

Une convolution est un produit scalaire local entre :
- un patch de l’image
- un filtre (kernel)

Formule simplifiée :

```
S(i,j) = Σ (I ⊙ K)
```

### 4.2 Apprentissage des filtres

Les filtres sont appris par descente de gradient.

---

## 5. Global Average Pooling (GAP)

### 5.1 Rôle

Réduit les cartes de caractéristiques en vecteur.

### 5.2 Formule

```
GAP_c = (1 / HW) Σ feature_map_c
```

---

## 6. Couche Dense

Chaque neurone calcule :

```
y = W · x + b
```

---

## 7. Batch Normalization

```
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β
```

---

## 8. ReLU

```
ReLU(x) = max(0, x)
```

---

## 9. Softmax

```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

---

## 10. Cross-Entropy

```
Loss = -log(p_y)
```

---

## 11. Backpropagation

```
W_new = W_old - η × gradient
```

---

## 12. Transfer Learning

- Phase 1 : backbone gelé
- Phase 2 : fine-tuning

---

## 13. Régularisation

- Dropout
- Class Weights

---

## 14. Évaluation

- Accuracy
- Matrice de confusion
- F1-score

---

## 15. TensorFlow Lite

- Optimisation DEFAULT
- Quantification

---

## 16. Flutter

Respect strict du prétraitement et des labels.

---

## 17. Conclusion

De la théorie au déploiement mobile.

---

✍️ Claud-IA · 2025
