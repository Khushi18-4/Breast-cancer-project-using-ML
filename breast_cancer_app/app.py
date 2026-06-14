from flask import Flask, render_template, request
import numpy as np
import sklearn.datasets


import breast_cancer_machine_learning_project as ml_project 

app = Flask(__name__)

# Features name
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
feature_names = breast_cancer_dataset.feature_names

@app.route('/')
def home():
    return render_template('index.html', features=feature_names, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Taking 30 inputs data from frontend
            input_features = [float(request.form[name]) for name in feature_names]
            input_data_as_numpy_array = np.asarray(input_features)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            
           # calling the trained 'model' from ml_project
            prediction = ml_project.model.predict(input_data_reshaped)
            
            if prediction[0] == 0:
                result = "Malignant (Cancer Detected - Please consult a doctor)"
                alert_type = "danger"
            else:
                result = "Benign (No Cancer Detected - Patient is safe)"
                alert_type = "success"
                
            return render_template('index.html', 
                                   features=feature_names, 
                                   prediction_text=result, 
                                   alert_type=alert_type,
                                   form_data=request.form)
        except Exception as e:
            return render_template('index.html', features=feature_names, prediction_text=f"Error: {str(e)}", alert_type="warning")

if __name__ == "__main__":
    app.run(debug=True)