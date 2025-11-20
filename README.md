**NSL-KDD Intrusion Detection System (IDS) using Neural Networks**
=====================================================================

Un projet complet basé sur **Deep Learning** qui utilise la base NSL-KDD pour détecter les attaques réseau.

Le projet est divisé en **3 notebooks** :

📁 **Structure du projet**
=============================

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ├── data/  │   ├── X.npy  │   ├── y.npy  │  ├── models/  │   ├── scaler.save  │   ├── nslkdd_dnn_model.h5  │   ├── training_history.npy  │  ├── notebooks/  │   ├── 01_preprocessing.ipynb  │   ├── 02_training.ipynb  │   ├── 03_inference_visualization.ipynb  │  └── README.md   `

**Objectif du projet**
=========================

Construire un **IDS (Intrusion Detection System)** capable de classer un trafic réseau en :

*   **Normal (0)**
    
*   **Attack (1)**
    

Le modèle utilise un **réseau de neurones profond (DNN)**.

**Notebook 1 – Préprocessing**
=================================

Dans ce notebook :

### ✔ Chargement des données

NSL-KDD Train+Test

### ✔ Nettoyage

*   Suppression de la colonne vide
    
*   Correction des noms de colonnes
    
*   Conversion en labels binaires
    

### ✔ Sélection des 18 features importantes

### ✔ Encodage

one-hot pour service et flag

### ✔ Normalisation

StandardScaler → **sauvegardé dans /models**

### ✔ Sauvegarde du dataset préprocessé

*   X.npy
    
*   y.npy
    

**Notebook 2 – Entraînement du modèle**
==========================================

### ✔ Découpage train / test

train\_test\_split(stratify=y)

### ✔ Définition d’un DNN

*   Dense(64, relu)
    
*   Dense(32, relu)
    
*   Dropout
    
*   Sortie sigmoïde (binaire)
    

### ✔ EarlyStopping

Évite le sur-apprentissage

### ✔ Entraînement et évaluation

Affichage :

*   Accuracy
    
*   Precision
    
*   Recall
    
*   F1-score
    
*   Matrice de confusion
    

### ✔ Sauvegarde du modèle entraîné

models/nslkdd\_dnn\_model.h5

### ✔ Sauvegarde de l’historique

training\_history.npy

**Notebook 3 – Inférence + Visualisation**
=============================================

Ce notebook recharge :

*   Le modèle
    
*   Le scaler
    
*   Les données préprocessées
    
*   L’historique
    

Il affiche :

### ✔ Matrice de confusion (heatmap)

### ✔ Courbes d’accuracy

### ✔ Courbes de loss

### ✔ Prédictions du modèle

C’est ici que se fait la **visualisation finale**.

🛠 Installation
===============

### 1️⃣ Créer un environnement Python 3.10

(TensorFlow ne fonctionne pas sur Python 3.12)

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   conda create -n tf310 python=3.10  conda activate tf310   `

### 2️⃣ Installer les dépendances

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install numpy pandas scikit-learn tensorflow matplotlib seaborn joblib   `

### 3️⃣ Lancer Jupyter Notebook

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   jupyter notebook   `

Commande pour tester le modèle
=================================

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   model.predict(X_test[0].reshape(1, -1))   `

Retourne :

*   **≥ 0.5 → Attack**
    
*   **< 0.5 → Normal**
    

**Technologies utilisées**
=============================

*   Python
    
*   TensorFlow / Keras
    
*   Scikit-learn
    
*   Pandas
    
*   Matplotlib
    
*   Seaborn
    



**3\. Commit message pour ce README + Notebook 3**
====================================================
`   git add README.md notebooks/inference_visualization.ipynb  git commit -m "Added Notebook 3 (Inference & Visualization) + Full README documentation"  git push origin your-branch   `