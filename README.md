# Traffic_Signal_Detector

This project implements a deep learning-based system to recognize traffic signs using a Convolutional Neural Network (CNN), and perform real-time detection using OpenCV. The system is trained on a labeled dataset of traffic sign images, and the trained model is integrated with a live webcam feed to detect and classify signs in real time. It serves as a foundational prototype for intelligent transportation systems, including driver assistance and autonomous navigation.

---

## Project Overview

The project consists of two major components:

- **Traffic Sign Classification**: A CNN model trained on images of traffic signs to accurately classify them into predefined categories such as Stop, No Entry, Speed Limit, and more.
- **Live Detection**: Integration of the trained model with OpenCV to recognize and display traffic sign predictions in real time through a webcam.

The combination of image classification and live inference showcases how machine learning and computer vision can be merged for real-world safety-critical applications.

---

##  Dataset Information
The dataset consists of **42 different traffic sign classes**, representing a diverse set of road signs (e.g., warning, regulatory, and informational signs). The images are organized into class-wise folders, and a `labels.csv` file maps each class index to its corresponding traffic sign name.

---

##  Model Architecture

The classification model is a custom Convolutional Neural Network built using TensorFlow/Keras. Key components include:

- **Input Layer**: 32×32 grayscale images
- **Convolutional Layers**: Two convolution blocks with filters of size 5×5 and 3×3, each followed by ReLU activation
- **Pooling Layers**: MaxPooling layers to downsample feature maps
- **Dropout Layers**: Applied after convolution blocks and dense layers to prevent overfitting
- **Fully Connected Layers**: Dense layers to interpret features
- **Output Layer**: Softmax layer for multiclass classification

**Compilation:**
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (learning rate: 0.001)
- **Metrics**: Accuracy

---

##  Output & Evaluation

- The model is trained over multiple epochs and evaluated on training, validation, and test splits.
- Accuracy and loss trends are plotted to visualize performance over time.
- Final **test accuracy** is printed after evaluation on unseen data.
- The trained model is saved as a `.h5` file for later use in the real-time detection system.

---

##  Real-Time Detection

The live detection system uses a webcam feed to:
- Capture frames in real time
- Apply the same preprocessing as training
- Use the trained CNN model to classify traffic signs on-screen
- Display predictions overlayed on the video feed

This allows for quick validation of the model’s practical utility and responsiveness.

---

##  Applications

- Driver Assistance Systems
- Intelligent Transportation Solutions
- Autonomous Vehicle Prototyping
- Computer Vision & ML Research

