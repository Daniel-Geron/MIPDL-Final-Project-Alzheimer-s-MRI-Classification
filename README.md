# Alzheimer's Disease Stage Classification from MRI Scans

**Authors:** Daniel Geron & Amit Cohen  
**Institution:** Ruppin Academic Center, Department of Computer Science  
**Course:** Medical Image Processing and Deep Learning (MIPDL)

---

## 📌 Project Overview
This repository contains an automated, end-to-end deep learning pipeline designed to classify the progression stages of Alzheimer's disease from Magnetic Resonance Imaging (MRI) scans. Early and accurate diagnosis of Alzheimer's is critical for effective intervention, and distinguishing between subtle neurodegenerative stages manually is a complex, error-prone task.

Our model automatically classifies an input MRI scan into one of four diagnostic stages:
1. **Non-Demented**
2. **Very Mild Demented**
3. **Mild Demented**
4. **Moderate Demented**

By leveraging Transfer Learning via the **ResNet50** architecture, coupled with advanced Hyperparameter Optimization (HPO) and targeted fine-tuning, the final model achieves a mathematically unbiased **97.34% diagnostic accuracy** on a strictly quarantined test set.

## 🗂️ Repository Structure
* `MIPDL_Project_ResNet50_Daniel_Geron_Amit_Cohen.ipynb`: The primary, fully annotated Google Colab notebook containing the end-to-end pipeline (Data Ingestion, Preprocessing, HPO, ResNet50 Training, Fine-Tuning, and Evaluation).
* `ResNet18_Experiment.ipynb` *(or your exact file name)*: An auxiliary notebook detailing our preliminary architectural experiments using the lighter **ResNet18** model. This experiment served as our baseline (yielding 75.0% accuracy) and justified the necessity of upgrading to the deeper ResNet50 network to minimize false negatives in medical diagnostics.
* `README.md`: Project documentation.

## ⚙️ Methodology & Pipeline
1. **Data Ingestion & Automation:** Dynamic extraction of a 44,000-image skull-stripped MRI dataset directly into the cloud environment.
2. **Standardization:** Rigorous dimensional correction using **bicubic interpolation** to preserve microscopic spatial hierarchies, and matrix-level color-space redundancy processing (safely expanding Grayscale to RGB without data loss).
3. **Hyperparameter Optimization (HPO):** Automated Random Search tuning using a custom medical evaluation metric formula: `Score = (0.5 × Recall) + (0.5 × Accuracy) - (0.1 × Loss)`.
4. **Two-Phase Training:** * *Phase 1:* Feature extraction using a frozen ResNet50 base and a custom Global Average Pooling (GAP) head.
   * *Phase 2:* Microscopic fine-tuning by unthawing the 32-layer `conv5` block to adapt pre-trained filters specifically to MRI textures.

## 📊 Key Results
* **Baseline ResNet18:** 75.0% Accuracy
* **Baseline ResNet50 (Pre-Tuning):** 80.0% Accuracy
* **ResNet50 (After HPO):** 93.0% Accuracy
* **ResNet50 (After Fine-Tuning):** 98.0% Validation Accuracy
* **Final Unbiased Test Set Performance:** **97.34% Accuracy** (with 0.97 Recall for "Very Mild Demented" cases and near-perfect AUC scores across all classes).

## 🛠️ Technical Stack & Libraries
This project is built using industry-standard Python libraries for deep learning and computer vision:
* **TensorFlow & Keras:** Foundational deep learning frameworks for building, compiling, and training the ResNet50 architecture, utilizing mixed-precision GPU acceleration and asynchronous data prefetching.
* **Keras Tuner:** Deployed for automated Hyperparameter Optimization (HPO) across randomized model permutations.
* **OpenCV (`cv2`) & PIL:** Used for the rigorous image preprocessing pipeline, specifically executing bicubic interpolation and analyzing structural color-space redundancy.
* **Pandas & NumPy:** Core libraries for Exploratory Data Analysis (EDA) and rapid, vectorized matrix-level validations.
* **Scikit-Learn:** Utilized for statistical data partitioning (stratified splits) and generating advanced classification metrics (Precision, Recall, F1-Score, Confusion Matrices, ROC/AUC).
* **Matplotlib & Seaborn:** Integrated to dynamically plot empirical visualizations, including learning curves and heatmap confusion matrices.

## 🚀 How to Run
The code is explicitly engineered to be completely independent and self-contained. 
1. Open the primary `.ipynb` file in **Google Colab**.
2. Ensure the Runtime is set to **GPU** (`Runtime` > `Change runtime type` > `T4 GPU` or higher).
3. Select `Run All`. The script will automatically fetch the dataset from the cloud, install required dependencies, and execute the entire pipeline from top to bottom.
