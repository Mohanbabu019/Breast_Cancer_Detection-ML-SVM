from flask import Flask, render_template, request
import pickle
import numpy as np

#loading svm 
filename = 'breast_cancer.pkl'
with open(filename,'rb') as f:
     model=pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Make sure you have an index.html file in a templates folder

@app.route('/predict', methods=['POST'])
def predict():

    rad = float(request.form['Radius'])
    text = float(request.form['Texture'])
    area = float(request.form['Area'])
    smt = float(request.form['Smoothness'])
    compact = float(request.form['Compactness'])
    conca = float(request.form['Concavity'])
    concave = float(request.form['Concave'])
    symm = float(request.form['symmetry'])
    frac = float(request.form['fractal_dimension'])
    # Prepare the data for prediction
    data = np.array([[rad, text, area, smt, compact, conca, concave, symm, frac]])
    
    # Make prediction
    my_prediction = model.predict(data)
    
    # Assuming binary classification, you can adjust this as needed
    result = "Malignant" if my_prediction[0] == 1 else "Benign"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
