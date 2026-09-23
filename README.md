# CNN Classifier for Li-ion Battery Thermal Images

Experimental deep-learning workflow for binary classification of lithium-ion battery thermal images using a convolutional neural network implemented with TensorFlow/Keras.

## Objective

Explore whether image-based thermal patterns can distinguish two battery thermal-condition classes:

- normal
- high-temperature

This repository is part of broader work involving **lithium-ion batteries, thermal imaging, computer vision, and machine learning**.

## Architecture

- Conv2D: 32 filters
- MaxPooling2D + Dropout
- Conv2D: 64 filters
- MaxPooling2D + Dropout
- Conv2D: 128 filters
- MaxPooling2D + Dropout
- Dense: 256 units
- Sigmoid output for binary classification
- L2 regularization
- Adam optimizer
- Binary cross-entropy loss

## Tech stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Google Colab

## Repository contents

- `cnn_classifier.py` — model definition, training and single-image inference
- `requirements.txt` — main Python dependencies

The original research dataset is **not included** in this public repository.

## Research context

The project belongs to ongoing work in battery thermal analysis and data-driven classification. It is intended as a compact public portfolio example rather than a release of the complete doctoral research dataset or workflow.

## Author

**Salvador Durán Carmona**  
M.Sc. Mechanical Engineering | Ph.D. Candidate in Advanced Technology  
CICATA Querétaro, Instituto Politécnico Nacional (IPN), Mexico
