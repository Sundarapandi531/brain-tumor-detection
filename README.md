# Brain Tumor Classification using ResNet101

This project focuses on classifying MRI brain scans into four categories:

`Glioma Tumor`
Meningioma Tumor
Pituitary Tumor
No Tumor (Normal)

We utilize the ResNet101 deep learning model to achieve high accuracy in detecting and classifying brain tumors from MRI images.

📂 Dataset

The dataset contains MRI images of size 512x512, which are resized to 256x256 before training.
It consists of the following folders:

Training set: Images used for training the model

Testing set: Images used for evaluating model performance

Classes:

glioma/
meningioma/
pituitary/
normal/

⚙️ Technologies Used

Python 3
TensorFlow / Keras
ResNet101 (Transfer Learning)
Matplotlib, NumPy, Pandas

Google Colab / Kaggle for training

🚀 Model Architecture

Base Model: Pre-trained ResNet101 (ImageNet weights)
Custom Layers: Global Average Pooling, Dense, Dropout, Softmax for classification
Image Size: 256x256
Optimizer: Adam

Loss Function: Categorical Crossentropy

📊 Results

Training Accuracy: ~XX%
Validation Accuracy: ~XX%
Model is capable of distinguishing between different tumor types effectively.
