import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 1. LOAD DATA
# Ensure your data collection/preprocessing script created these
try:
    X = np.load("X.npy") 
    y = np.load("y.npy")
except FileNotFoundError:
    print("❌ ERROR: X.npy or y.npy not found. Run your data preprocessing first!")
    exit()

DATA_DIR = "data/raw"

# --- AUTO-SYNC LABELS ---
# Scans all folders (e.g., APPLE, BANANA, INVALID, WHAT...)
actions = sorted([label for label in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, label))])
actions = np.array(actions)
num_classes = len(actions)

# --- SHAPE SETTINGS ---
n_samples, total_frames, n_features = X.shape 
NEW_SEQ_LEN = 10  # Synced with your 30-frame collection

# Crop X to exactly 30 frames
X_final = X[:, :NEW_SEQ_LEN, :]

print(f"📦 Training Shape: {X_final.shape}") 
print(f"🏷️ Labels Detected ({num_classes}): {actions}")

# 2. BUILD MODEL (Enhanced for 20-word complexity)
model = Sequential([
    # Layer 1: Captures initial movement
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(NEW_SEQ_LEN, n_features)),
    Dropout(0.3),
    
    # Layer 2: Deeper understanding of the sign
    Bidirectional(LSTM(128, return_sequences=True)), 
    Dropout(0.3),
    
    # Layer 3: Final temporal summary
    Bidirectional(LSTM(64)),
    
    # Dense Layers for classification
    Dense(64, activation='relu'),
    Dense(32, activation='relu'), 
    
    # OUTPUT: Matches your folder count (e.g., 20)
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# --- 3. TRAINING SETTINGS ---
num_epochs = 150  # Fixed the 'NameError' by defining it here
batch_size = 32

# Callbacks to prevent overfitting and help the model learn
lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.00001)
early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

# --- 4. TRAIN ---
print(f"🚀 Training model for {num_epochs} epochs...")

model.fit(
    X_final, 
    y, 
    epochs=num_epochs,      
    batch_size=batch_size, 
    shuffle=True,    
    callbacks=[lr_reducer, early_stop]
)

# --- 5. SAVE EVERYTHING ---
if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/sign_model.h5")
np.save("models/labels.npy", actions)

print("---")
print("✅ FINAL 20-WORD MODEL SAVED to 'models/sign_model.h5'")