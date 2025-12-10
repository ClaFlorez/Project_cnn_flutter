# EfficientNetB3 pour la Classification d’Images d’Animaux  
### Modèle CNN haute précision optimisé pour le déploiement mobile  
**Autrice : Claudia (Claud-IA)**  
**Version : 2025**

---

## 📄 Résumé (Abstract)

Ce projet présente un modèle de classification d’images d’animaux basé sur **EfficientNetB3** avec fine-tuning partiel, atteignant une précision d’environ **99 %** sur un jeu de test équilibré (6 classes : éléphant, girafe, léopard, rhinocéros, tigre, zèbre).

Plusieurs backbones ont été comparés, dont *MobileNetV2* et *EfficientNetB0*, et un ablation study a été réalisé pour analyser l'effet des différents hyperparamètres (learning rate, freeze ratio, dropout, class weights).

Le modèle final est exporté en **TensorFlow Lite (TFLite)** et validé pour une intégration en temps réel sur une application Flutter.

---

## 1. Introduction

La classification d’images est une tâche fondamentale en vision par ordinateur.  
Grâce au transfert d’apprentissage, les modèles pré-entraînés comme EfficientNet permettent d’obtenir des performances très élevées même avec un entraînement limité.

Objectifs du projet :

- Comparer différents backbones CNN légers  
- Optimiser la précision via fine-tuning  
- Réduire la confusion entre classes proches (ex. rhinocéros vs éléphant)  
- Exporter un modèle TFLite pour une utilisation mobile  
- Créer un pipeline reproductible et documenté

---

## 2. Description du Jeu de Données

### 2.1 Classes
- elephant  
- girafe  
- leopard  
- rhino  
- tigre  
- zebre  

### 2.2 Répartition

| Split       | Nombre d’images |
|-------------|------------------|
| Entraînement | 20 400           |
| Validation   | 3 600            |
| Test         | 6 000            |

### 2.3 Organisation

