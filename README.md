# Brain_tumar_classification_project

# 🧠 Brain Tumor MRI Classification Using Deep Learning

## 📌 Project Overview

This project focuses on classifying Brain MRI scans into four categories using Transfer Learning and Deep Learning techniques.

The objective is to assist in the automatic detection of brain tumors from MRI images by leveraging powerful pretrained Convolutional Neural Networks (CNNs).

### Classes

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

---

## 📂 Dataset

Dataset used:

**Brain Tumor MRI Dataset (Kaggle)**

The dataset contains MRI images organized into training and testing folders.

### Dataset Structure

```text
Training/
│
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/

Testing/
│
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

---

## 🚀 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Google Colab

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Image resizing to **224 × 224**
* Dataset batching
* Train-validation split (80%-20%)
* Data shuffling
* Prefetching using TensorFlow AUTOTUNE
* Label inference from directory structure

---

## 🏗️ Deep Learning Models

Three pretrained CNN architectures were evaluated:

### 1. EfficientNetB0

* Lightweight and efficient architecture
* Excellent accuracy-to-parameter ratio
* Selected as the final model

### 2. ResNet101

* Deep residual network
* Handles vanishing gradient problems effectively

### 3. DenseNet121

* Dense feature connections
* Efficient feature reuse

---

## ⚙️ Transfer Learning

Pretrained ImageNet weights were used.

```python
weights='imagenet'
include_top=False
```

The convolutional base was frozen:

```python
base_model.trainable = False
```

Custom classification layers were added on top:

* GlobalAveragePooling2D
* Dense(256, ReLU)
* Dropout(0.3)
* Dense(256, ReLU)
* Dropout(0.3)
* Dense(4, Softmax)

---

## 🛑 Overfitting Prevention

The following techniques were used:

* Transfer Learning
* Dropout Layers
* Early Stopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

---

## 📊 Model Evaluation

Evaluation metrics:

* Accuracy
* Validation Accuracy
* Loss
* Validation Loss
* Confusion Matrix

Training and validation performance were visualized using Matplotlib.

---

## 📈 Model Comparison

The performance of the following models was compared:

| Model          | Status          |
| -------------- | --------------- |
| EfficientNetB0 | Best Performing |
| ResNet101      | Evaluated       |
| DenseNet121    | Evaluated       |

EfficientNetB0 achieved the best overall performance and was selected as the final model.

---

## 📊 Results

### Model Comparison

| Model          | Accuracy   |
| -------------- | ---------- |
| EfficientNetB0 | **90.81%** |
| ResNet101      | **82.81%** |
| DenseNet121    | **70.88%** |

### Best Performing Model

EfficientNetB0 achieved the highest accuracy of **90.81%** and outperformed ResNet101 and DenseNet121 on the Brain MRI classification task.

### Key Findings

* EfficientNetB0 provided the best balance between performance and computational efficiency.
* ResNet101 achieved competitive performance but remained below EfficientNetB0.
* DenseNet121 showed lower classification accuracy on this dataset.
* Transfer Learning significantly improved model performance compared to training from scratch.

### Final Model

✅ Selected Model: **EfficientNetB0**

✅ Accuracy: **90.81%**

✅ Number of Classes: **4**

* Glioma
* Meningioma
* Pituitary
* No Tumor

### efficient_net accuracy vs val_accuracy
<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/b1ea7920-07f9-4bd4-a734-53db2878ee43" />


### efficient_net loss vs val_loss
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/812e12e6-799a-4edc-abe5-556d79b71ddb" />


---

## 💾 Final Model

The trained model was saved as:

```text
brain_tumor_model.keras
```

Load model:

```python
from tensorflow.keras.models import load_model

model = load_model("brain_tumor_model.keras")
```

---

## 🔮 Prediction Example

```python
prediction = model.predict(image)
predicted_class = class_names[np.argmax(prediction)]
```

The model predicts one of the four brain MRI categories along with confidence scores.

---

## 📁 Project Structure

```text
Brain-Tumor-Classification/
│
├── BRAIN_TUMOR.ipynb
├── brain_tumor_model.keras
├── README.md
└── images/
```

---

## 🎯 Future Improvements

* Streamlit Web Application
* Model Fine-Tuning
* Explainable AI (Grad-CAM)
* MRI Report Generation
* Cloud Deployment

---

## 👨‍💻 Author

**Kartik Jain**

B.Tech (Information Technology)

Deep Learning | Machine Learning | Computer Vision
