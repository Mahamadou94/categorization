from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model_file.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Convert data into DataFrame 
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Generate recommendation based on prediction
    recommendation = 'Accept' if prediction == 1 else 'Reject' 
    
    return jsonify({'recommendation': recommendation})

if __name__ == '__main__':
    app.run(debug=True)
