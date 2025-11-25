# **Intrusion Detection System using NSL-KDD**

This project implements a **Deep Learning–based Intrusion Detection System (IDS)** using the **NSL-KDD dataset**. It predicts whether a network connection is **Normal** or an **Attack** based on selected network traffic features and provides **explainable AI insights** using **LIME**.

---

## **Project Description**

The **NSL-KDD dataset** provides training and testing data for **network attack classification**. It is widely used in **Intrusion Detection Systems (IDS)** to detect suspicious activities in computer networks.

The dataset has **41 features** grouped into three categories:

1. **Basic Features**
2. **Content Features**
3. **Traffic Features**

Each record represents a network connection and contains:
✔ Feature values
✔ A label: **Normal** or **Attack**

---

## **Project Objective**

The goal of this project is to:

### 1️⃣ **Select the Most Important Features**

Using **Information Gain (IG)**, **18 significant features** were selected to:

* Reduce dataset dimensionality
* Improve training speed
* Keep the most discriminative features for classification

**Selected features:**

```
service, flag, src_bytes, dst_bytes, logged_in, count, 
serror_rate, srv_serror_rate, same_srv_rate, diff_srv_rate, 
dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, 
dst_host_diff_srv_rate, dst_host_same_src_port_rate, 
dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate
```

---

### 2️⃣ **Classification Method: Multilayer Perceptron (MLP)**

A **Multilayer Perceptron (MLP)** is used to classify connections into:

* **Normal (0)**
* **Attack (1)**

**Why MLP?**

✔ Learns nonlinear relationships between features
✔ Handles encoded and normalized features efficiently
✔ Provides strong performance for network security tasks

**Model Architecture:**

* Dense(64) — ReLU
* Dropout(0.3)
* Dense(32) — ReLU
* Dropout(0.2)
* Dense(1) — Sigmoid

---

## **Project Steps**

### **1) Data Preprocessing (`Preprocessing.ipynb`)**

✔ Remove unnecessary columns
✔ Binary labeling: *normal → 0, attack → 1*
✔ Select 18 key features
✔ One-hot encode categorical variables
✔ Normalize features using `StandardScaler`
✔ Save preprocessed data + scaler

### **2) MLP Training (`Training.ipynb`)**

✔ Train/test split
✔ Model construction
✔ Early stopping
✔ Save trained model

### **3) Evaluation & Visualization (`Evaluation_visualization.ipynb`)**

✔ Confusion matrix
✔ Accuracy
✔ Learning curves
✔ Performance analysis

### **4) Explainable AI with LIME (`Explainable_AI.ipynb`)**

✔ Generate local explanations for model predictions
✔ Identify feature contributions to each prediction
✔ Visualize feature importance per sample

---

## **Key Results**

* **Overall Accuracy:** ~93–98%
* **Effective detection** of common attacks (DoS, Probe)
* **Low false positives** for normal traffic
* Confusion matrix shows **clear separation** between Normal and Attack

**LIME explanations** help understand **why the model predicts attacks**, improving transparency.

---

## **Conclusion**

This project demonstrates that using:

* **Information Gain feature selection**
* **Deep Neural Networks (MLP)**
* **Explainable AI (LIME)**

enables:

✔ Automatic intrusion detection
✔ Reduced dataset complexity
✔ Reliable binary classification
✔ Transparent AI-based IDS

---

## **Project Structure**

```
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   ├── X.npy
│   └── y.npy
│
├── models/
│   ├── scaler.save
│   ├── training_history.npy
│   └── nslkdd_dnn_model.h5
│
├── notebooks/
│   ├── Preprocessing.ipynb
│   ├── Training.ipynb
│   ├── Evaluation_visualization.ipynb
│   └── Explainable_AI.ipynb
├── app.py
└── README.md
```

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/HattourWejden/nsl-kdd-intrusion-detection.git
cd NSL_KDD
```

2. Create a Python environment:

```bash
conda create -n nsl_kdd python=3.10
conda activate nsl_kdd
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Required libraries:** `streamlit`, `tensorflow`, `pandas`, `numpy`, `joblib`, `matplotlib`, `scikit-learn`, `lime`

---

## **Usage**

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Enter network connection features manually or choose **predefined samples** (Normal, DoS, Probe)
2. Click **Predict** to see:

   * **Prediction (Normal/Attack)**
   * **Confidence score (%)**
3. Visual explanations of predictions are generated using **LIME**

---

## **Example Input / Output**

| Feature           | Normal | DoS Attack |
| ----------------- | ------ | ---------- |
| `src_bytes`       | 215    | 0          |
| `dst_bytes`       | 4500   | 0          |
| `logged_in`       | 1      | 0          |
| `count`           | 5      | 200        |
| `serror_rate`     | 0.0    | 1.0        |
| `srv_serror_rate` | 0.0    | 1.0        |
| `same_srv_rate`   | 0.7    | 0.0        |
| `diff_srv_rate`   | 0.1    | 1.0        |

---

## **References**

* NSL-KDD Dataset: [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)
* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
* LIME Explainable AI: [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)
