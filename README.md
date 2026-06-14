# Breast-cancer-project-using-ML
A website that utilizes Flask and Machine Learning to predict breast cancer. Built on the Wisconsin dataset using Logistic Regression, it processes 30 features to classify tumors as Benign or Malignant. Features a modern, responsive UI with Tailwind CSS.
# Breast Cancer Detection Web App

This is a visually attractive and interactive web application that uses a Machine Learning model (**Logistic Regression**) to predict whether a breast tumor is **Malignant** or **Benign** based on 30 cellular features.

## 📁 Project Structure
- `breast_cancer_machine_learning_project.py`: Core ML training script.
- `app.py`: Flask backend connecting frontend with the ML model.
- `templates/index.html`: Responsive Tailwind CSS frontend UI.

## 🛠️ How to run locally
1. Clone this repository or download the zip.
2. Open the folder in VS Code.
3. Install required libraries:
   ```bash
   pip install flask numpy pandas scikit-learn
