# Indian Currency Detection Model

## Overview

This project presents a Python-based model for detecting and recognizing Indian currency notes using computer vision and machine learning techniques. The model is designed to assist in the automated identification of currency denominations from images, providing a valuable tool for financial institutions, retail businesses, and visually impaired individuals. 

## Features

- **High Accuracy**: Utilizes state-of-the-art deep learning algorithms to achieve high accuracy in detecting various denominations of Indian currency notes.
- **Robust to Variations**: The model is robust against variations in lighting, angle, and partial occlusion, making it suitable for real-world applications.
- **Real-Time Processing**: Optimized for real-time processing on standard computing hardware.
- **Scalable and Extensible**: Designed to be easily extended to recognize additional denominations or adapt to different currencies.

## Technical Details

### Model Architecture

The currency detection model employs a convolutional neural network (CNN) architecture, leveraging transfer learning from pre-trained models (e.g., VGG16, ResNet50). This approach allows the model to learn discriminative features specific to Indian currency notes while benefiting from the generalized knowledge embedded in pre-trained networks.

### Data Collection and Preprocessing

- **Data Collection**: The training dataset comprises thousands of images of Indian currency notes captured under various conditions. Images were sourced from public datasets and augmented using techniques such as rotation, scaling, and noise addition to improve generalization.
- **Preprocessing**: Before feeding images into the model, preprocessing steps such as resizing, normalization, and contrast enhancement are applied to ensure uniformity and improve model performance.

### Training and Optimization

- **Training**: The model is trained using supervised learning with labeled images of different denominations. The training process involves backpropagation with stochastic gradient descent (SGD) and data augmentation strategies to enhance robustness.
- **Optimization**: Techniques such as dropout, batch normalization, and learning rate scheduling are employed to prevent overfitting and improve convergence.

## Installation

To run the currency detection model, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hidayat-Ali/currency-detection.git
   cd currency-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**:
   ```bash
   python detect_currency.py --image_path ./sample_images/note.jpg
   ```

## Results

The model demonstrates impressive accuracy across a range of denominations, consistently achieving over 95% accuracy on a held-out test set. It also shows resilience to common real-world challenges such as variations in lighting and partial obstructions.

## Applications and Future Work

This currency detection model has significant potential for various applications, including:

- **Assistive Technology**: Enhancing accessibility tools for visually impaired individuals by enabling them to identify currency notes easily.
- **Banking and Retail**: Automating the cash handling process in banks and retail stores, reducing errors, and increasing efficiency.
- **Educational Use**: Serving as a practical example of applying machine learning in computer vision tasks for educational purposes.

### Future Directions

We are actively exploring several avenues to enhance and expand the capabilities of this model:

- **Multi-Currency Detection**: Expanding the model to recognize currency notes from other countries.
- **Integration with Mobile Platforms**: Developing a mobile application for on-the-go currency detection.
- **Incorporation of Edge AI**: Implementing model optimization techniques for deployment on edge devices like smartphones and embedded systems.

