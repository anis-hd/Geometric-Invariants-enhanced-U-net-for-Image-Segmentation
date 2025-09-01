# app.py

import os
import glob
import tkinter as tk
from tkinter import filedialog
import eventlet
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename


#importing created classes
from trainer import Trainer
from benchmarker import Benchmarker
from geometricInvariants import InvariantCalculator
from inference import InferenceRunner


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key!'
socketio = SocketIO(app, async_mode='eventlet')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


@app.route('/')
def index():
    """Renders the main interface."""
    return render_template('index.html')


@app.route('/preview_data', methods=['POST'])
def preview_data():
    try:
        data = request.json
        image_dir = data.get('image_dir')
        mask_dir = data.get('mask_dir')
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            return jsonify({'error': 'One or more directories not found.'}), 400
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:5]
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))[:5]
        return jsonify({'images': image_files, 'masks': mask_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_image')
def get_image():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "File not found", 404
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in allowed_extensions):
        return send_file(path)
    else:
        return "Access denied", 403


@app.route('/browse-folder')
def browse_folder():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(master=root, title="Select a Folder")
    root.destroy()
    return jsonify({'path': folder_path})


@app.route('/browse-file')
def browse_file():
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        master=root,
        title="Select a File",
        filetypes=[("All files", "*.*"), ("CSV files", "*.csv"), ("Image files", "*.jpg *.png")]
    )
    root.destroy()
    return jsonify({'path': file_path})


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        return jsonify({'path': save_path})


# SOCKET-IO EVENT HANDLER

@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('start_training')
def handle_start_training(config):
    print('Received training config:', config)
    required_paths = ['image_dir', 'mask_dir', 'class_csv', 'output_dir']
    if any(not config.get(path) for path in required_paths):
        socketio.emit('training_error', {'error': 'Missing required path. Please fill out all path fields.'})
        return

    try:
        # Basic parameters
        config['epochs'] = int(config['epochs']); config['batch_size'] = int(config['batch_size'])
        config['learning_rate'] = float(config['learning_rate']); config['img_size'] = int(config['img_size'])
        config['data_subset'] = int(config['data_subset'])
        
        # Advanced architecture parameters
        config['unet_features'] = [int(f.strip()) for f in config['unet_features'].split(',') if f.strip()]
        config['film_hidden_dim'] = int(config['film_hidden_dim'])
        config['patch_hidden_dim'] = int(config['patch_hidden_dim'])

    except (ValueError, KeyError) as e:
        error_msg = f'Invalid hyperparameter format. Please check your inputs. Error: {e}'
        socketio.emit('training_error', {'error': error_msg})
        return
    
    trainer_instance = Trainer(config, socketio)
    socketio.start_background_task(trainer_instance.run)


@socketio.on('start_benchmarking')
def handle_start_benchmarking(config):
    print('Received benchmarking config:', config)
    required_paths = ['class_csv', 'output_dir']
    if any(not config.get(path) for path in required_paths):
        error_msg = "Missing required path(s). Please fill out the fields on the 'Benchmark' tab."
        socketio.emit('benchmark_error', {'error': error_msg})
        return
        
    config['batch_size'] = int(config['batch_size'])
    
    output_dir = config['output_dir']
    config['processed_image_dir'] = os.path.join(output_dir, "processed_images")
    config['processed_mask_dir'] = os.path.join(output_dir, "processed_masks")
    config['cache_dir'] = os.path.join(output_dir, "cache")
    config['saved_models_dir'] = os.path.join(output_dir, "saved_models")
    
    benchmarker_instance = Benchmarker(config, socketio)
    socketio.start_background_task(benchmarker_instance.run)


@socketio.on('start_inference')
def handle_start_inference(config):
    print('Received inference config:', config)
    required_paths = ['class_csv', 'output_dir', 'image_path']
    if any(not config.get(path) for path in required_paths):
        socketio.emit('inference_error', {'error': 'Missing required path(s). Please fill out all fields.'})
        return
        
    # No need to parse advanced params; they are loaded from the config file on the backend.
    inference_instance = InferenceRunner(config, socketio)
    socketio.start_background_task(inference_instance.run)


@socketio.on('calculate_invariants')
def handle_calculate_invariants(config):
    print('Received invariant calculation config:', config)
    if not config.get('image_path') or not config.get('method'):
        socketio.emit('invariant_error', {'error': 'Missing image path or calculation method.'})
        return
        
    calculator_instance = InvariantCalculator(config, socketio)
    socketio.start_background_task(calculator_instance.run)


if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)