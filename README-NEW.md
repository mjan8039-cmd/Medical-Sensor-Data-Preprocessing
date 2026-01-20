Medical Sensor Data Preprocessing Pipeline

Project Overview
This project implements an original and enhanced preprocessing pipeline for numerical medical sensor data. The work is inspired by a 2023 IEEE Access research paper on wearable sensor-based cardiovascular abnormality detection. The preprocessing steps are redesigned to improve data quality, robustness, and model readiness.

 Selected Research Paper
Title: An Explainable Deep Learning Framework for Wearable Sensor–Based Early Detection of Cardiovascular Abnormalities  
Authors: Y. Zhang et al.  
Venue: IEEE Access  
Year: 2023  

Dataset
- Source: UCI Machine Learning Repository  
- Dataset: Heart Rate Variability (HRV) Dataset  
- Type: Numerical physiological sensor data  
- Samples: ~3000  
- Features: 20+ numerical features  

Preprocessing Steps
1. Missing value handling using interpolation and median imputation
2. Outlier detection using Isolation Forest
3. Feature scaling using PowerTransformer (Yeo-Johnson)
4. Noise reduction using rolling window smoothing
5. Dimensionality reduction using PCA
6. Train-test split (80/20)

Technologies Used
- Python
- pandas
- NumPy
- scikit-learn

pip install -r requirements.txt
python src/preprocessing_pipeline.py
