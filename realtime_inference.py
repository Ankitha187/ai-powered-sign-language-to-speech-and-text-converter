import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from collections import deque
import os

app = Flask(__name__)
# Standard SocketIO setup for real-time bridge
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 1. MANUAL LABELS ---
# This matches your current 'HOW' and 'YOU' data folders
labels = ["HOW", "YOU"] 

# --- 2. LOAD YOUR TRAINED MODEL ---
try:
    # Ensure this path matches where your .h5 file is stored
    model = load_model("models/sign_model.h5")
    print(f"✅ BRIDGE ACTIVE. Recognizing: {labels}")
except Exception as e:
    print(f"❌ MODEL ERROR: {e}")

# --- 3. LOGIC SETTINGS ---
EXPECTED_FEATURES = 39  # Your model expects 39 features
SEQUENCE_LENGTH = 2    # Your model expects 2 frames (None, 2, 39)
sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence_list = []

@app.route('/')
def index():
    return render_template('index.html')

# --- 4. THE HAND LANDMARK BRIDGE ---
@socketio.on('landmarks')
def handle_landmarks(data):
    global sequence, sentence_list
    try:
        # Convert incoming 63 values to the 39 your model needs
        features = np.array(data)[:EXPECTED_FEATURES]
        sequence.append(features)

        # Only predict once we have exactly 2 frames
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(list(sequence), axis=0)
            res = model.predict(input_data, verbose=0)[0]
            
            idx = np.argmax(res)
            confidence = res[idx]
            
            # Debug: See what the AI is thinking in the VS Code Terminal
            # print(f"AI Sees: {labels[idx]} ({confidence:.2f})")

            # Only send to website if the AI is very sure (> 85%)
            if confidence > 0.85:
                # Security check: ensure the index exists in our labels list
                if idx < len(labels):
                    predicted_word = labels[idx]
                    
                    # Logic: Don't repeat the same word twice in a row
                    if not sentence_list or predicted_word != sentence_list[-1]:
                        sentence_list.append(predicted_word)
                        
                        # Send result back to the browser
                        socketio.emit('prediction', {
                            'word': predicted_word,
                            'sentence': " ".join(sentence_list)
                        })
    except Exception as e:
        print(f"Processing Error: {e}")

# --- 5. THE CLEAR BUTTON LISTENER ---
@socketio.on('clear_sentence')
def handle_clear():
    global sentence_list
    sentence_list = []
    print("🗑️ Sentence Cleared via UI")

if __name__ == '__main__':
    # Using debug=True helps you see errors in the terminal immediately
    socketio.run(app, debug=True, port=5000)