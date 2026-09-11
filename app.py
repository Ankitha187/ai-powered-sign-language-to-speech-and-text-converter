import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from collections import deque
import os
import sys

# Ensure feature_extraction.py is in the same folder
try:
    from feature_extraction import extract_features 
except ImportError:
    print("❌ ERROR: feature_extraction.py not found in this folder!")
    sys.exit()

app = Flask(__name__)
# High-speed WebSocket configuration
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# --- 1. DYNAMIC FOLDER SCANNER ---
DATA_DIR = "data/raw"
if os.path.exists(DATA_DIR):
    # Automatically finds all categories and sorts them alphabetically
    labels = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])
    print(f"✅ Bridge synced with {len(labels)} categories: {labels}")
else:
    labels = ["HOW", "YOU", "INVALID"] # Fallback
    print("⚠️ Warning: data/raw not found. Using defaults.")

# --- 2. LOAD AI MODEL ---
MODEL_PATH = "models/sign_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ AI Model Loaded Successfully.")
else:
    model = None
    print(f"⚠️ ERROR: {MODEL_PATH} not found. AI will not predict.")

SEQUENCE_LENGTH = 10 
sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence_list = []

# Mock class for MediaPipe compatibility
class Landmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

# --- UPDATED ROUTING ---

@app.route('/')
def landing_page():
    # This serves your new high-end landing page (main.html)
    return render_template('main.html')

@app.route('/translator')
def translator_page():
    # This serves your working camera code (index.html)
    return render_template('index.html')

# --- 3. REAL-TIME DATA PROCESSING ---
@socketio.on('landmarks')
def handle_landmarks(data):
    global sequence, sentence_list
    if model is None: return

    try:
        # Reconstruct coordinates from the web list
        reconstructed = [Landmark(data[i], data[i+1], data[i+2]) for i in range(0, len(data), 3)]

        # Extract features (must match your training math)
        features = extract_features(reconstructed)
        sequence.append(features)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(list(sequence), axis=0)
            res = model.predict(input_data, verbose=0)[0]
            idx = np.argmax(res)
            confidence = res[idx]

            # Threshold check
            if confidence > 0.85:
                predicted_word = labels[idx].upper()
                
                # --- INVALID SIGN LOGIC ---
                if "INVALID" in predicted_word:
                    socketio.emit('prediction', {
                        'word': "INVALID HAND SIGN",
                        'sentence': " ".join(sentence_list),
                        'status': 'warning'
                    })
                    return 
                
                # --- NEW WORD LOGIC ---
                if not sentence_list or predicted_word != sentence_list[-1]:
                    sentence_list.append(predicted_word)
                    socketio.emit('prediction', {
                        'word': predicted_word,
                        'sentence': " ".join(sentence_list),
                        'status': 'success'
                    })
    except Exception as e:
        print(f"❌ Processing Error: {e}")

# --- 4. CONTROL HANDLERS (BACKSPACE & CLEAR) ---

@socketio.on('backspace')
def handle_backspace():
    global sentence_list, sequence
    if len(sentence_list) > 0:
        removed_word = sentence_list.pop() # Removes the last word
        sequence.clear() # Clears AI memory to prevent immediate re-detection
        print(f"⌫ Removed: {removed_word}")
        
        socketio.emit('prediction', {
            'word': "REMOVED",
            'sentence': " ".join(sentence_list),
            'status': 'success'
        })

@socketio.on('clear_sentence')
def handle_clear():
    global sentence_list, sequence
    sentence_list = []
    sequence.clear() # Fully reset the AI sequence
    print("🧹 UI and Sequence Cleared")

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)