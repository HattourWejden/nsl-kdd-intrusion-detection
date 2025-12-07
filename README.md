# Intrusion Detection System using NSL-KDD

This project implements a **Deep Learning–based Intrusion Detection System (IDS)** using the **NSL-KDD dataset**.  
The system predicts whether a network connection is **Normal** or an **Attack** (binary classification) and provides **explainable AI** insights using **LIME**. A **Streamlit web app** is included for interactive testing.

---

## 1. Project Description

Traditional rule-based IDS struggle to detect **novel or complex attacks** and are difficult to maintain. This project uses a **Multi-Layer Perceptron (MLP)** trained on the NSL-KDD dataset to automatically learn patterns of malicious vs normal traffic.

Key ideas:

- Use **Information Gain** to select the most informative features from NSL-KDD.
- Train a **DNN/MLP** for binary classification:  
  `0 → Normal`, `1 → Attack`
- Build a **visual evaluation pipeline** (confusion matrix, learning curves).
- Add **LIME** to explain individual predictions (what features triggered an “Attack”).

---

## 2. Dataset: NSL-KDD

The project uses:

- `KDDTrain+.txt` – training data
- `KDDTest+.txt` – test data

Each record describes a network connection with:

- **Numerical features** (e.g., `src_bytes`, `dst_bytes`, `count`, `serror_rate`, etc.)
- **Categorical features** (e.g., `protocol_type`, `service`, `flag`)
- **Label** (normal or specific attack type)

In this project, all attack types are **grouped into a single “Attack” class** for binary classification.

---

## 3. Project Objective

- Build a **binary IDS classifier**: Normal vs Attack.
- Achieve **high accuracy** and **low false positives** on the NSL-KDD test set.
- Provide **transparent explanations** using LIME so a security analyst can understand **why** the model flagged a connection as an attack.

---

## 4. Project Steps

### 4.1. Data Preprocessing (`notebooks/Preprocessing.ipynb`)

Main operations:

1. **Load raw NSL-KDD data**

   ```text
   data/nsl-kdd/KDDTrain+.txt
   data/nsl-kdd/KDDTest+.txt
   ```

2. **Drop unnecessary columns**  
   Remove features that don’t improve performance or are redundant.

3. **Binary labeling**

   - `normal` → label `0`
   - any attack type (DoS, Probe, R2L, U2R, etc.) → label `1`

4. **Feature selection**

   - Select ~18 most important features based on **Information Gain** or prior analysis.

5. **Encode categorical variables**

   - Apply **one-hot encoding** to fields like `service` and `flag`.

6. **Feature scaling**

   - Use `StandardScaler` from scikit-learn to normalize numerical features.

7. **Save preprocessed artifacts**

   - `data/X.npy` – feature matrix
   - `data/y.npy` – labels
   - `models/scaler.save` – fitted scaler used later in training and in the app

The notebook ends with:

```text
✔ Preprocessing done!
Saved:
 -> data/X.npy
 -> data/y.npy
 -> models/scaler.save
```

---

### 4.2. Model Training (`notebooks/Training.ipynb`)

1. **Load preprocessed data**

   ```python
   X = np.load("data/X.npy")
   y = np.load("data/y.npy")
   ```

   Example shapes from the notebook:

   ```text
   X shape: (148517, 95)
   y shape: (148517,)
   ```

2. **Train/test split**

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

3. **Model architecture (MLP / DNN)**

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Dropout

   model = Sequential([
       Dense(64, activation='relu', input_shape=(input_dim,)),
       Dropout(0.3),

       Dense(32, activation='relu'),
       Dropout(0.2),

       Dense(1, activation='sigmoid')  # binary output
   ])

   model.compile(
       optimizer='adam',
       loss='binary_crossentropy',
       metrics=['accuracy']
   )
   ```

4. **Regularization with Early Stopping**

   ```python
   from tensorflow.keras.callbacks import EarlyStopping

   early_stop = EarlyStopping(
       monitor='val_loss',
       patience=5,
       restore_best_weights=True
   )

   history = model.fit(
       X_train, y_train,
       validation_split=0.2,
       epochs=30,
       batch_size=64,
       callbacks=[early_stop],
       verbose=1
   )
   ```

5. **Evaluation on test set**

   ```python
   from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

   y_pred = (model.predict(X_test) > 0.5).astype(int)

   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("Classification Report:")
   print(classification_report(y_test, y_pred))

   print("Confusion Matrix:")
   print(confusion_matrix(y_test, y_pred))
   ```

   Example reported performance:

   ```text
   Accuracy: ~0.9865 (≈ 98.7%)

   Classification Report:
                 precision    recall  f1-score   support

              0       0.99      0.99      0.99     15411
              1       0.99      0.99      0.99     14293

       accuracy                           0.99     29704
      macro avg       0.99      0.99      0.99     29704
   weighted avg       0.99      0.99      0.99     29704

   Confusion Matrix:
   [[15223   188]
    [  211 14082]]
   ```

   This shows:

   - **Very high accuracy (~99%)**
   - **Balanced precision/recall** for both classes
   - **Low false positives** and **low false negatives**

6. **Save model and training history**

   ```python
   model.save("models/nslkdd_dnn_model.keras")
   np.save("models/training_history.npy", history.history)
   ```

---

### 4.3. Evaluation & Visualization (`notebooks/Evaluation_visualisation.ipynb`)

This notebook:

1. **Loads data and trained model**

   ```python
   X = np.load("../data/X.npy")
   y = np.load("../data/y.npy")
   model = load_model("../models/nslkdd_dnn_model.keras")
   ```

2. **Computes predictions and metrics**

   ```python
   from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

   y_pred = (model.predict(X) > 0.5).astype(int)
   cm = confusion_matrix(y, y_pred)
   print(classification_report(y, y_pred))
   print("Accuracy:", accuracy_score(y, y_pred))
   ```

3. **Plots confusion matrix**

   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   plt.figure(figsize=(6,5))
   sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
   plt.title("Confusion Matrix")
   plt.xlabel("Predicted")
   plt.ylabel("True")
   plt.show()
   ```

4. **Plots learning curves**

   ```python
   history = np.load("../models/training_history.npy", allow_pickle=True).item()

   plt.figure(figsize=(12,5))

   # Accuracy
   plt.subplot(1,2,1)
   plt.plot(history['accuracy'], label='Train')
   plt.plot(history['val_accuracy'], label='Validation')
   plt.title("Accuracy Curve")
   plt.xlabel("Epochs")
   plt.ylabel("Accuracy")
   plt.legend()

   # Loss
   plt.subplot(1,2,2)
   plt.plot(history['loss'], label='Train')
   plt.plot(history['val_loss'], label='Validation')
   plt.title("Loss Curve")
   plt.xlabel("Epochs")
   plt.ylabel("Loss")
   plt.legend()

   plt.show()
   ```

These plots help detect **overfitting/underfitting** and confirm model stability.

---

### 4.4. Explainable AI with LIME (`notebooks/Explainable_AI.ipynb`)

Objective: **Explain individual model predictions**.

1. **Load processed data and model**

   ```python
   import numpy as np
   import joblib
   from lime.lime_tabular import LimeTabularExplainer
   from tensorflow.keras.models import load_model

   X = np.load("../data/X.npy")
   scaler = joblib.load("../models/scaler.save")
   model = load_model("../models/nslkdd_dnn_model.keras")
   ```

2. **Configure LIME explainer**

   ```python
   explainer = LimeTabularExplainer(
       training_data=X,
       feature_names=[...],          # list of selected feature names
       class_names=["Normal", "Attack"],
       mode='classification'
   )
   ```

3. **Explain a single instance**

   ```python
   i = 0  # index of sample to explain
   exp = explainer.explain_instance(
       X[i],
       model.predict,
       num_features=10
   )
   exp.show_in_notebook()
   ```

LIME outputs a **bar plot** showing which features **increase** or **decrease** the probability of “Attack”. This makes the IDS **more transparent** for analysts.

---

## 5. Streamlit Application (`app.py`)

The Streamlit app provides an easy UI to test the IDS.

### 5.1. Features

- Manual input for selected features (e.g., `src_bytes`, `dst_bytes`, `count`, `serror_rate`, etc.).
- Sidebar with **predefined examples** (e.g., “Normal” / “DoS attack” / “Probe attack”).
- Real-time **prediction** with:
  - Label: **Normal** / **Attack**
  - **Confidence score** (%)
- (Optionally) integration with LIME for **explanations**.

Example snippet from the app:

```python
st.title("🛡️ Intrusion Detection System – NSL-KDD")
st.markdown("### Detect whether a network connection is **Normal** or an **Attack**")

sample_inputs = {
    "Normal": {
        "src_bytes": 181, "dst_bytes": 5450, "logged_in": 1,
        "count": 2, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
        "dst_host_count": 150, "dst_host_srv_count": 150,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.5, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "service": "http", "flag": "SF"
    },
    ...
}

selected_sample = st.sidebar.selectbox("Select example", ["None"] + list(sample_inputs.keys()))
if selected_sample != "None":
    form_data = sample_inputs[selected_sample]
    st.sidebar.info(f"Example loaded: {selected_sample}")
else:
    form_data = sample_inputs["Normal"]
```

The app:

1. Collects input features.
2. Applies the **same preprocessing pipeline** (encoding + scaling).
3. Uses the trained MLP to predict.
4. Displays prediction and confidence.

---

## 6. Key Results

From training and evaluation notebooks:

- **Overall accuracy**: ~**98–99%** on the test set.
- **High precision and recall** for both classes.
- **Confusion matrix**:

  - Very few normal connections misclassified as attacks.
  - Very few attacks misclassified as normal.

- **Learning curves** show **stable training** with no severe overfitting (thanks to dropout and early stopping).

With **LIME**, you can see which features (e.g., `serror_rate`, `dst_bytes`, `logged_in`) are most influential for each prediction, making the IDS **interpretable**.

---

## 7. Project Structure

```text
├── data/
│   ├── nsl-kdd/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   ├── X.npy          # preprocessed features
│   └── y.npy          # labels
│
├── models/
│   ├── scaler.save             # StandardScaler
│   ├── training_history.npy    # training/validation metrics per epoch
│   └── nslkdd_dnn_model.keras     # trained MLP model
│
├── notebooks/
│   ├── Preprocessing.ipynb
│   ├── Training.ipynb
│   ├── Evaluation_visualisation.ipynb
│   └── Explainable_AI.ipynb
│
├── app.py          # Streamlit web interface
└── README.md
```

---

## 8. Installation

1. **Clone the repository**

```bash
git clone https://github.com/HattourWejden/nsl-kdd-intrusion-detection.git
cd nsl-kdd-intrusion-detection
```

2. **Create a Python environment**

```bash
conda create -n nsl_kdd python=3.10
conda activate nsl_kdd
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Required main libraries:

- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`
- `lime`
- `joblib`

---

## 9. Usage

### 9.1. Re-run the pipeline (optional)

1. **Preprocessing**

   Open and run `notebooks/Preprocessing.ipynb` to regenerate `X.npy`, `y.npy`, and `scaler.save`.

2. **Training**

   Run `notebooks/Training.ipynb` to train the MLP and save `nslkdd_dnn_model.keras` and `training_history.npy`.

3. **Evaluation**

   Run `notebooks/Evaluation_visualisation.ipynb` to see evaluation metrics and learning curves.

4. **Explanations**

   Run `notebooks/Explainable_AI.ipynb` to generate LIME explanations for specific samples.

### 9.2. Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL in your browser (usually `http://localhost:8501`), select or enter the connection features, and click **Predict**.

---

## 10. Example: Model Output Interpretation

Given a sample connection, the model returns:

- **Prediction**: `Attack`
- **Probability**: `0.97` (97%)

LIME may show that the most influential features are:

- `serror_rate` (high)
- `srv_serror_rate` (high)
- `dst_bytes` (low)
- `logged_in = 0`

This suggests that the model considers **frequent SYN errors**, **no successful login**, and **unusual byte patterns** as strong indicators of an attack.
