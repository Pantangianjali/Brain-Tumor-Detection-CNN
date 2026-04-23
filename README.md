# Brain Tumor Detection using CNN

Deep learning model to classify brain MRI images into tumor vs no-tumor using Convolutional Neural Networks.

## Results
- **Validation Accuracy**: 96.8%
- **Model**: Custom CNN with 4 Conv2D layers + Data Augmentation
- **Dataset**: Brain MRI Images from Kaggle

## Tech Stack
Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Scikit-learn

## Key Features
1. **Data Preprocessing**: Resized images to 224x224, normalized pixel values, data augmentation
2. **CNN Architecture**: 4 Conv2D + MaxPooling layers, Dropout for regularization
3. **Training**: Adam optimizer, Binary crossentropy loss, Early stopping
4. **Evaluation**: Achieved 96.8% validation accuracy

## How to Run
1. Clone repo: `git clone https://github.com/Pantanglanjali/Brain-Tumor-Detection-CNN`
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset from Kaggle and place in `dataset/` folder
4. Run: `python main.py`

## Dataset
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
