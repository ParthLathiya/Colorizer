
import os
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from source.usemodel import UNetColorizer, load_model, colorize_image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MODEL_PATH = "results 5e16b/checkpoints/best_model.pth"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")
model = load_model(MODEL_PATH, device)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process image
            gray_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                 return jsonify({'error': 'Could not read image'}), 400

            colorized = colorize_image(gray_img, model, device)
            
            result_filename = f"colored_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, colorized)
            
            return jsonify({
                'original': url_for('static', filename=f'uploads/{filename}'),
                'result': url_for('static', filename=f'results/{result_filename}')
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
