
# Intrusion Detection Using Hybrid Deep Learning Techniques

## 🚀 Project Overview
Intrusion Detection Systems (IDS) play a critical role in maintaining the security of networks by identifying malicious activities. Traditional IDS solutions struggle with high false-positive rates and often fail to detect sophisticated, evolving threats. This project proposes a **Hybrid Deep Learning-Based IDS** that utilizes **Autoencoders**, **CNNs (Convolutional Neural Networks)**, and **LSTMs (Long Short-Term Memory)** to achieve high accuracy in detecting and classifying network intrusions.

### Key Features:
- **Hybrid Architecture**: Combines autoencoders for feature extraction with CNN-LSTM for spatial-temporal analysis.
- **Multi-class Detection**: Identifies a range of network anomalies, including DoS, DDoS, and probing attacks.
- **Scalability**: Designed to handle extensive datasets and real-time traffic.
- **User Interface**: Includes a browser-based interface for real-time analysis and visualization.

---

## 📜 Table of Contents
1. [Introduction](#-introduction)
2. [Installation](#-installation)
3. [Usage](#-usage)
4. [Dependencies](#-dependencies)
5. [Methodology](#-methodology)
6. [Results](#-results)
7. [Contributing](#-contributing)
8. [Future Enhancements](#-future-enhancements)
9. [License](#-license)

---

## 🌟 Introduction
The increasing sophistication of cyber threats has highlighted the need for intelligent, adaptive, and real-time IDS solutions. By integrating deep learning techniques, this project aims to enhance intrusion detection by:
- Extracting meaningful features with **autoencoders**.
- Analyzing spatial features using **CNN layers**.
- Capturing temporal dependencies with **LSTM layers**.

This hybrid approach reduces false positives and ensures robust anomaly detection in diverse network environments.

---

## 🛠 Installation
To set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/nids-hybrid-deep-learning.git
   cd nids-hybrid-deep-learning
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scriptsctivate     # For Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   - Use publicly available datasets like [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) or [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).
   - Place the dataset in the `data/` directory.

5. **Run Preprocessing Scripts**:
   ```bash
   python src/preprocess_data.py
   ```

6. **Train the Model**:
   ```bash
   python src/train_model.py
   ```

---

## 📖 Usage
After setting up, you can use the project for both training and inference:

1. **Train the Model**:
   ```bash
   python src/train_model.py --epochs 50 --batch_size 32
   ```

2. **Evaluate the Model**:
   ```bash
   python src/evaluate_model.py --model_path models/hybrid_model.h5
   ```

3. **Run the Real-Time Analyzer**:
   ```bash
   python src/run_analyzer.py
   ```
   Access the browser-based interface at [http://localhost:5000](http://localhost:5000).

---

## 📦 Dependencies
Ensure the following libraries are installed:
- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Flask (for the web interface)
- Jinja2 (templating for the web interface)

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🔬 Methodology
1. **Data Preprocessing**:
   - Normalize features using MinMaxScaler.
   - Remove outliers and handle missing data.

2. **Feature Extraction**:
   - Utilize an autoencoder to reduce the dimensionality and noise of the dataset.

3. **Hybrid Model**:
   - **CNN Layers**: Extract spatial patterns in network traffic.
   - **LSTM Layers**: Capture temporal sequences for anomaly detection.

4. **Evaluation Metrics**:
   - **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

---

## 📊 Results
- **Accuracy**: Achieved up to 95% accuracy in multi-class classification.
- **False Positive Rate**: Reduced false positives significantly compared to traditional models.
- **Visualization**: Detailed graphs for loss, accuracy, and confusion matrices are available in the `results/` folder.

---

## 🤝 Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes and push to your fork.
4. Create a Pull Request.

---

## 🚀 Future Enhancements
- Integrate additional datasets for improved generalization.
- Optimize the model for real-time deployment.
- Add support for containerized deployment using Docker.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
