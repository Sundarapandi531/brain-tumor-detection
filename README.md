# 🧠 Brain Tumor Classification using ResNet101

This project focuses on classifying MRI brain scans into four categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor (Normal)**

We utilize the **ResNet101 deep learning model** to achieve high accuracy in detecting and classifying brain tumors from MRI images.

---

## 📂 Dataset

- The dataset contains MRI images of size **512x512**, which are **resized to 256x256** before training.
- Dataset structure:

```
dataset/
│── Training/
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ └── normal/
│
│── Testing/
├── glioma/
├── meningioma/
├── pituitary/
└── normal/  
```

---

## ⚙️ Technologies Used

- Python 3  
- TensorFlow / Keras  
- ResNet101 (Transfer Learning)  
- Matplotlib, NumPy, Pandas  
- Google Colab / Kaggle for training  

---

## 🚀 Model Architecture

- **Base Model:** Pre-trained **ResNet101** (ImageNet weights)  
- **Custom Layers:**  
  - Global Average Pooling  
  - Dense Layers  
  - Dropout  
  - Softmax for classification  

- **Image Size:** `256x256`  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  

---

## 📊 Results

- **Training Accuracy:** ~89%  
- **Validation Accuracy:** ~87%  
- The model is capable of distinguishing between different tumor types effectively.  

---
