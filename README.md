# 🧠 Denoising Convolutional Autoencoder avec Grad-CAM

## 📌 Description du Projet

Ce projet implémente un **autoencodeur convolutionnel profond** pour supprimer le bruit des images du dataset **Fashion-MNIST**.  
L’objectif est de reconstruire des images propres à partir d’images bruitées et d’évaluer la qualité des reconstructions avec des métriques quantitatives (MSE, PSNR, SSIM).  
Le projet inclut également une **visualisation Grad-CAM** pour comprendre quelles zones de l’image influencent la reconstruction.

---

## 📌 Technologies Utilisées

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- scikit-image  

---

## 📌 Préparation des Données

- Importation des bibliothèques nécessaires (NumPy, TensorFlow/Keras, Matplotlib, métriques PSNR/SSIM).  
- Chargement du dataset **Fashion-MNIST** (60 000 images d’entraînement, 10 000 images de test).  
- Normalisation des images entre 0 et 1.  
- Adaptation au format CNN `(28, 28, 1)`.  
- Les images originales sont utilisées comme cibles (autoencodeur).

---

## 📌 Augmentation des Données et Ajout de Bruit

- Définition d’une **augmentation de données** : rotations et translations pour enrichir le dataset.  
- Création de générateurs pour l’entraînement et la validation (`ImageDataGenerator`).  
- Ajout d’un **bruit gaussien** (`noise_factor = 0.2`) pour simuler des images bruitées.  
- Générateurs `train_noisy_generator` et `val_noisy_generator` fournissant les images bruitées et leurs cibles originales.

---

## 📌 Architecture de l’Autoencodeur

- Construit pour **supprimer le bruit** des images Fashion-MNIST.  
- **Encodeur** : 7 couches (Conv2D + MaxPooling2D) compressant l’information dans la **bottleneck layer**.  
- **Décodeur** : 8 couches (Conv2D + UpSampling2D) pour reconstruire l’image originale.  
- Activation finale **sigmoid** pour obtenir des pixels entre 0 et 1.  
- Compilation avec **Adam**, perte **MSE**, métrique **SSIM**.

---

## 📌 Entraînement

- Le modèle est entraîné sur les images bruitées avec **validation sur le jeu de test**.  
- **EarlyStopping** utilisé : surveille la `val_loss`, patience = 5 epochs, restaure les meilleurs poids.  
- Entraînement sur **20 epochs maximum**, batch_size = 128.  
- Historique (`history`) enregistre la **loss (MSE)** et la **métrique SSIM** pour chaque epoch.

---

## 📌 Affichage des Courbes d’Apprentissage

- Visualisation de la **perte (MSE)** pour l’entraînement et la validation.  
- Visualisation de la **métrique SSIM** pour l’entraînement et la validation.  
- Permet de suivre la performance du modèle au fil des epochs et de détecter un éventuel overfitting.

---

## 📌 Grad-CAM & Visualisation

- Génération d’**images bruitées** pour visualisation.  
- Application de **Grad-CAM** sur la couche bottleneck pour identifier les zones importantes du modèle.  
- Affichage côte à côte de :  
  - l’image originale  
  - l’image bruitée  
  - l’image reconstruite  
  - la **carte de chaleur Grad-CAM** superposée.  
- Permet de comprendre **quelles parties de l’image influencent le débruitage**.

---

## 📌 Évaluation et Résultats

- Évaluation sur un échantillon du jeu de test.  
- Calcul des métriques :  
  - **MSE** (Erreur Quadratique Moyenne) ↓  
  - **PSNR** (Peak Signal-to-Noise Ratio) ↑  
  - **SSIM** (Structural Similarity Index) ↑  
- Ces métriques permettent de mesurer la qualité de reconstruction et la similarité avec les images originales.

---

## 📌 Améliorations Futures

- Tester sur des images plus grandes ou en couleur.  
- Comparer avec d’autres architectures (U-Net, VAE).  
- Déploiement du modèle pour une interface interactive.  
- Optimisation de l’autoencodeur pour des reconstructions plus rapides et plus précises.

---

## 📌 Références

- Dataset : [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)  
- TensorFlow / Keras documentation  
- Grad-CAM : [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
