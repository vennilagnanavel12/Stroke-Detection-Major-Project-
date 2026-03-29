This project focuses on detecting strokes from medical imaging data using Deep Learning techniques.
It leverages the power of TensorFlow and the pre-trained VGG-16 architecture to classify brain images as stroke or non-stroke.
Early detection of stroke is critical in medical diagnosis, and this model aims to assist healthcare professionals by providing accurate and fast predictions.

Project Structure:
stroke-detection/
│── dataset/
│   ├── train/
│   ├── test/
│
│── models/
│   └── vgg16_model.h5
│
│── notebooks/
│   └── training.ipynb
│
│── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│
│── requirements.txt
│── README.md


📊 Results
Training Accuracy: ~XX%
Validation Accuracy: ~XX%
Loss Graphs and Accuracy Plots available in /notebooks
notebooks

📸 Sample Output
Input: Brain MRI Image
Output:
🟢 No Stroke
🔴 Stroke Detected

