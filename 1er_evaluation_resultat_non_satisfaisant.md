# 📊 Rapport expliqué – Évaluation du modèle CNN (classification d’animaux)

Ce document explique en détail le rapport d’évaluation généré pour mon modèle **CNN** utilisé dans le projet *Project_cnn_flutter* (reconnaissance d’animaux dans des images).

L’objectif est que on puisses **comprendre chaque chiffre** du rapport et **le réutiliser tel quel dans ton GitHub / documentation de projet**.

---

## 1. Contexte général du modèle

- **Type de modèle** : Réseau de neurones convolutif (CNN)
- **Tâche** : Classification d’images en **6 classes d’animaux**  
  - `elephant`, `girafe`, `leopard`, `rhino`, `tigre`, `zebre`
- **Fichier du modèle évalué** :  
  `/content/drive/MyDrive/models/cnn_model_20251205_160914_best.h5`
- **Date et heure de l’évaluation** :  
  `2025-12-07 14:53:44`

Ce modèle est probablement celui que tu utilises ensuite dans ton application **Flutter + TFLite**, donc ce rapport correspond **à la qualité réelle** du modèle que tu prévois d’intégrer dans l’app.

---

## 2. Configuration du jeu de test

Dans la section **« 1. CONFIGURATION »**, on retrouve les paramètres de l’évaluation :

- **Données de test** :  
  `/content/Project_cnn_flutter/test`
- **Nombre total d’images de test** : `600`
- **Nombre de classes** : `6`
- **Répartition (supposée équilibrée)** :  
  → environ **100 images par classe** (`600 / 6`)
- **Taille des images** : `224 x 224`
- **Batch size** : `32`

🔎 **Interprétation :**

- 600 images, c’est un **jeu de test raisonnable** pour avoir des statistiques relativement stables.
- 224×224 est une taille standard pour des CNN (compromis entre qualité visuelle et coût de calcul).
- Un batch size de 32 est classique pour l’évaluation.

---

## 3. Performance globale du modèle

Section **« 2. PERFORMANCE GLOBALE »** :

- **Accuracy** : `0.4750` → **47,50 %**
- **Précision (macro ou moyenne)** : `0.6250`
- **Recall (rappel)** : `0.4750`
- **F1-Score** : `0.4484`
- **Loss** : `1.3161`

### 3.1 Accuracy – 47,50 %

L’**accuracy** mesure la proportion de prédictions correctes :

\[
\text{Accuracy} = \frac{\text{nombre de prédictions correctes}}{\text{nombre total d’images}}
\]

Ici :  
- **Correctes** : 600 − 315 = **285** images  
- **Total** : 600 images  
- Accuracy ≈ 285 / 600 = 0,475 → **47,5 %**

➡️ Cela signifie que **le modèle se trompe encore sur un peu plus de la moitié des images**.  
Ce n’est **pas catastrophique** pour un premier modèle, mais **insuffisant** pour une application en production sans amélioration.

### 3.2 Précision – 0,6250

La **précision** mesure, parmi toutes les images **prédites** dans une classe, quelle proportion est correcte.

- Une précision de **0,6250** signifie qu’en moyenne :
  - Quand le modèle dit « *c’est un X* », il a raison **dans ~62,5 % des cas**.

Cela indique que le modèle est **relativement prudent** : quand il se prononce, c’est **plutôt fiable**, mais il fait encore beaucoup de confusions.

### 3.3 Recall (rappel) – 0,4750

Le **rappel** mesure, parmi toutes les images qui **sont réellement** d’une classe donnée, combien sont correctement détectées par le modèle.

- Un rappel de **0,4750** veut dire que le modèle ne détecte correctement que **47,5 %** des vrais exemples de chaque classe en moyenne.

➡️ Cela confirme que le modèle **manque** encore beaucoup d’animaux (il ne les reconnaît pas, ou les confond).

### 3.4 F1-Score – 0,4484

Le **F1-score** combine précision et rappel :

\[
F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}
\]

- Avec **Précision = 0,6250** et **Recall = 0,4750**, on obtient un F1 ≈ **0,4484**.
- Ce score reflète un **compromis moyen** : le modèle est un peu plus précis que sensible, mais globalement **modeste**.

### 3.5 Loss – 1,3161

La **loss** (fonction de coût) est une mesure interne du modèle (par ex. entropie croisée).

- Une loss de **1,3161** indique que le modèle est encore **loin de la situation idéale**.
- Ce chiffre est surtout utile pour **comparer plusieurs versions de modèle entre elles** (par exemple avant/après amélioration).

---

## 4. Performance par classe

Section **« 3. PERFORMANCE PAR CLASSE »**.

On dispose de métriques par classe :  
- **Nombre d’images**
- **Accuracy**
- **Précision**
- **Recall**
- **F1-Score**
- **AUC-ROC**
- **Average Precision (AP)**

Deux classes sont entièrement visibles dans le rapport :

### 4.1 Classe `elephant`

- Nombre d’images : **100**
- Accuracy (pour cette classe dans le test) : **0,3100**
- Précision : **0,5167**
- Recall : **0,3100**
- F1-Score : **0,3875**
- AUC-ROC : **0,8631**
- Average Precision : **0,5950**

🔎 **Interprétation :**

- Le modèle **ne détecte correctement que 31 %** des éléphants (rappel).
- Mais quand il prédit « elephant », il a raison **~51,7 %** du temps (précision).
- L’**AUC-ROC** est relativement bonne (0,86), ce qui signifie que les probabilités prédites contiennent **de l’information utile**, même si le seuil de décision ou l’entraînement global ne sont pas encore optimaux.

### 4.2 Classe `zebre`

- Nombre d’images : **100**
- Accuracy : **0,6600**
- Précision : **0,8462**
- Recall : **0,6600**
- F1-Score : **0,7416**
- AUC-ROC : **0,9377**
- Average Precision : **0,8098**

🔎 **Interprétation :**

- C’est l’une des **meilleures classes** du modèle :
  - Il trouve **66 %** des zèbres.
  - Quand il prédit « zebre », il a raison **dans ~84,6 %** des cas.
  - AUC-ROC très élevée (~0,94) → les zèbres sont **visuellement distinctifs** pour le modèle.

### 4.3 Autres classes (girafe, leopard, rhino, tigre)

Les valeurs exactes ne sont pas toutes visibles dans la sortie copiée, mais le rapport indique que :

- Certaines classes sont **nettement plus difficiles** que d’autres.
- Les **confusions fréquentes** (voir section suivante) montrent que :
  - Le modèle confond souvent **tigre ↔ girafe**,  
  - et **elephant / girafe / zebre ↔ rhino**.

En résumé :

- **zebre** : bien reconnu  
- **elephant** : moyen  
- **tigre / leopard / girafe / rhino** : plusieurs confusions importantes à corriger.

---

## 5. Analyse des erreurs

Section **« 4. ANALYSE DES ERREURS »** :

- **Nombre total d’erreurs** :  
  `315 / 600` → **52,50 %**
- **Erreurs à haute confiance (> 90 %)** : `0`
- **Prédictions incertaines mais correctes (< 60 %)** : `147`

🔎 **Ce que cela signifie :**

- Le modèle se trompe encore **plus d’une fois sur deux**.
- Cependant, il **ne fait pas d’erreurs extrêmement confiantes** (aucune erreur avec une confiance > 90 %).
  - C’est plutôt positif : quand il est **très sûr**, il ne se trompe quasiment pas.
- Il existe **147 cas où le modèle doute (confiance < 60 %) mais a raison** :
  - Ces cas pourraient être **intéressants pour ajuster le seuil de décision**, ou pour améliorer l’interface utilisateur (par ex. afficher un message « je ne suis pas sûr »).

### 5.1 Top 5 des confusions

Le rapport affiche :

1. `tigre → girafe` : **66 fois** (20,95 % des erreurs)
2. `elephant → rhino` : **58 fois** (18,41 %)
3. `leopard → girafe` : **56 fois** (17,78 %)
4. `girafe → rhino` : **21 fois** (6,67 %)
5. `zebre → rhino` : **16 fois** (5,08 %)

🔎 **Interprétation :**

- Le modèle **confond beaucoup** :
  - Les **tigres** avec des **girafes** (ce qui indique que le modèle ne capte pas bien les motifs/taches/rayures caractéristiques).
  - Les **éléphants, girafes et zèbres** avec des **rhinocéros** :
    - Cela suggère que certaines textures / couleurs / arrière-plans sont proches dans ton dataset.
- Cela peut venir :
  - de **photos trop similaires** entre classes,
  - de **bruit dans les données** (mauvaises étiquettes),
  - ou d’un modèle qui n’est pas encore assez **profond / régularisé / bien entraîné**.

---

## 6. Distribution des confiances

Section **« 5. DISTRIBUTION DES CONFIANCES »** :

- **Confiance moyenne (toutes les prédictions)** : `0.5358`
- **Confiance moyenne sur les prédictions correctes** : `0.5926`
- **Confiance moyenne sur les erreurs** : `0.4845`

🔎 **Interprétation :**

- En moyenne, le modèle donne des probabilités autour de **53 à 59 %**, donc il est **souvent incertain**.
- Les prédictions correctes ont une confiance **plus élevée** que les erreurs (0,59 vs 0,48), ce qui est **logique et sain** :
  - Cela montre que les probabilités contiennent une information utile pour **filtrer** les décisions (par exemple, ignorer les prédictions < 50 % dans l’app).

---

## 7. Fichiers générés et leur utilité

Section **« 6. FICHIERS GÉNÉRÉS »** :

1. `evaluation_..._classification_report.txt`  
   → Rapport texte détaillé (précision, rappel, F1 par classe).
2. `evaluation_..._confusion_matrix.png`  
   → Matrice de confusion visuelle (qui montre qui est confondu avec qui).
3. `evaluation_..._roc_curves.png`  
   → Courbes ROC par classe (sensibilité vs 1−spécificité).
4. `evaluation_..._precision_recall.png`  
   → Courbes précision–rappel par classe (très utiles si les classes sont déséquilibrées).
5. `evaluation_..._confidence_distribution.png`  
   → Histogramme de la distribution des confiances (toutes / correctes / erreurs).
6. `evaluation_..._prediction_examples.png`  
   → Exemples d’images avec la prédiction du modèle (utile pour analyser visuellement les erreurs).
7. `evaluation_..._full_report.txt`  
   → Le rapport complet que tu as généré (celui que nous expliquons ici).

🧩 **Intégration dans ton projet :**

- Ces fichiers sont parfaits pour :
  - ton **README technique**,
  - la **documentation GitHub**,
  - et une **présentation dans ta vidéo YouTube ou ton app Flutter**.

---

## 8. Conclusion et pistes d’amélioration

### 8.1 Bilan

- Le modèle **apprend quelque chose de réel** (les zèbres sont bien reconnus, les AUC sont souvent bonnes).
- Mais l’**accuracy globale de 47,5 %** et le nombre d’erreurs (315/600) montrent que :
  - le modèle est encore **trop limité pour un usage fiable**,
  - surtout à cause des **confusions entre certaines paires d’animaux**.

### 8.2 Idées d’amélioration

Voici quelques pistes pour améliorer ce modèle CNN :

1. **Augmentation de données**  
   - Rotation, zoom, recadrage, flip horizontal/vertical, changements de lumière, etc.
   - Cela aidera le modèle à mieux généraliser.

2. **Équilibrer / nettoyer le dataset**  
   - Vérifier qu’il y a bien **100 images de bonne qualité par classe**.  
   - Supprimer les images floues ou ambiguës.

3. **Architecture du modèle**  
   - Essayer un modèle un peu plus profond (plus de couches conv, batch norm, dropout).
   - Tester des modèles pré-entraînés (Transfer Learning) : MobileNet, EfficientNet, etc.

4. **Ajustement des seuils de décision**  
   - Utiliser la distribution des confiances pour :
     - fixer un seuil minimal (ex : ne pas accepter de prédictions < 0,6),
     - ou afficher une alerte « je ne suis pas sûr ».

5. **Analyse détaillée des images mal classées**  
   - Utiliser `prediction_examples.png` pour voir **visuellement** pourquoi le modèle se trompe :
     - arrière-plan confus ?
     - l’animal est trop petit dans l’image ?
     - animal partiellement caché ?

---

## 9. Résumé en une phrase 

> **Ce modèle CNN atteint une accuracy globale de 47,5 % sur 600 images de test (6 classes d’animaux), avec de bonnes performances sur les zèbres mais de fortes confusions entre tigres, girafes, éléphants et rhinocéros, ce qui en fait une base correcte pour un prototype Flutter + TFLite mais encore améliorable pour une utilisation en production.**

---
