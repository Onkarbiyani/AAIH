from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import io

# Import our new API function from inference.py
from inference import run_inference_api

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        # Read file as bytes
        image_bytes = file.read()
        
        # Run inference logic without dropping to disk
        # We ensure the model path maps correctly
        model_path = 'best_unet_model.pth'
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model weights not found. Train the model first!'}), 500
            
        results = run_inference_api(image_bytes, model_path=model_path)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
