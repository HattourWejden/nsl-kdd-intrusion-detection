# **Intrusion Detection System using NSL-KDD**

---
## **Description du Projet**

Le dataset **NSL-KDD** fournit un ensemble de données d’entraînement et de test destiné à la classification d’attaques réseau.
Il est utilisé dans les **Intrusion Detection Systems (IDS)** pour détecter les activités suspectes dans un système informatique.

Le format KDD comporte **41 features** réparties en 3 catégories :

1. **Basic Features**
2. **Content Features**
3. **Traffic Features**

Chaque enregistrement représente une connexion réseau et contient :
✔ un ensemble de caractéristiques
✔ un label correspondant à un comportement **normal** ou **malveillant**

---

## **Objectif du Travail**

Le but de ce projet est de :

### 1️⃣ **Sélectionner les caractéristiques les plus importantes**

À l’aide de la méthode **Information Gain (IG)**, 18 features significatives ont été retenues dans l’énoncé du projet afin de :

* réduire la dimensionnalité
* améliorer la vitesse d’entraînement
* conserver les features les plus discriminantes pour la classification

Les 18 features sélectionnées sont :

* service
* flag
* src_bytes
* dst_bytes
* logged_in
* count
* serror_rate
* srv_serror_rate
* same_srv_rate
* diff_srv_rate
* dst_host_count
* dst_host_srv_count
* dst_host_same_srv_rate
* dst_host_diff_srv_rate
* dst_host_same_src_port_rate
* dst_host_srv_diff_host_rate
* dst_host_serror_rate
* dst_host_srv_serror_rate

---

## **Méthode de Classification Choisie : Neural Network (MLP)**

Nous avons utilisé un **réseau de neurones multicouches (MLP – Multilayer Perceptron)** pour classer les connexions réseau en deux catégories :

* **Normal (0)**
* **Attack (1)**

Un MLP est particulièrement adapté aux données tabulaires comme NSL-KDD car :

✔ il apprend les relations non linéaires entre les features
✔ il gère très bien les features encodées et normalisées
✔ il obtient d’excellentes performances pour des tâches de sécurité réseau

### **Architecture du modèle**

* Dense(64) — ReLU
* Dropout(0.3)
* Dense(32) — ReLU
* Dropout(0.2)
* Dense(1) — Sigmoid

Cela correspond exactement à un IDS classique basé sur Deep Learning.

---

## **Étapes Réalisées**

### **1) Prétraitement des données (Preprocessing.ipynb)**

✔ Suppression des colonnes inutiles
✔ Étiquetage binaire :
    • *normal* → 0
    • *attaque* → 1
✔ Sélection des 18 features indiquées dans le sujet
✔ Encodage One-Hot des variables catégorielles
✔ Normalisation des données (StandardScaler)
✔ Sauvegarde du scaler + données prétraitées

### **2) Entraînement d’un MLP (Training.ipynb)**

✔ Split train/test
✔ Construction du modèle
✔ Early stopping
✔ Sauvegarde du modèle entraîné

### **3) Évaluation et Visualisation (Evaluation_visualisation.ipynb)**

✔ Matrice de confusion
✔ Accuracy
✔ Courbe d’apprentissage
✔ Analyse des performances

---

## **Résultats Principaux**

Après entraînement :

* **Accuracy globale :** ~93–97% (typique sur NSL-KDD)
* **Bonne détection des attaques fréquentes (DoS, Probe)**
* **Faible erreur sur le trafic normal**
* **Matrice de confusion montrant une nette séparation normal/attaque**

Ces résultats démontrent que le modèle MLP est **très efficace** pour identifier les comportements anormaux.

---

## **Conclusion**

Grâce à la sélection de features via **Information Gain** et à l’utilisation d’un **réseau de neurones**, ce projet permet de :

✔ détecter de manière automatique les intrusions
✔ réduire la complexité du dataset
✔ obtenir une classification binaire fiable
✔ mettre en place un IDS moderne basé sur l’IA

Le modèle obtenu peut constituer la base d’un :

* système de sécurité en entreprise
* firewall intelligent
* outil éducatif pour comprendre les attaques réseau
* prototype de système de détection d'intrusion en temps réel

---

## 📁 **Structure du Projet**

```
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   ├── X.npy
│   └── y.npy
│
├── models/
│   ├── scaler.save
│   └── nslkdd_model.h5
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   └── 03_visualization.ipynb
│
├── README.md
└── requirements.txt
```

---


