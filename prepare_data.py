import numpy as np
import os

DATA_DIR = "data/raw"
SEQUENCE_LENGTH = 30

# 1. Get only the folders (A, B, O, etc.)
# We use 'labels' here consistently
labels = [label for label in os.listdir(DATA_DIR)
          if os.path.isdir(os.path.join(DATA_DIR, label))]

labels = sorted(labels)
label_map = {label: idx for idx, label in enumerate(labels)}

X, y = [], []

print(f"Starting preparation for labels: {labels}")

for label in labels:
    label_path = os.path.join(DATA_DIR, label)
    files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
    
    for file in files:
        file_path = os.path.join(label_path, file)
        try:
            data = np.load(file_path)

            # ✅ If sequence is longer, trim it
            if data.shape[0] > SEQUENCE_LENGTH:
                data = data[:SEQUENCE_LENGTH]

            # ✅ If sequence is shorter, pad with zeros
            elif data.shape[0] < SEQUENCE_LENGTH:
                padding = np.zeros((SEQUENCE_LENGTH - data.shape[0], data.shape[1]))
                data = np.vstack((data, padding))

            X.append(data)
            y.append(label_map[label])
        except Exception as e:
            print(f"Skipping corrupted file {file}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# --- THE FIXES ---
# 1. Ensure we actually have data before saving
if len(X) > 0:
    np.save("X.npy", X)
    np.save("y.npy", y)
    
    # 2. Use 'labels' variable instead of 'actions' to avoid NameError
    # We convert it back to a numpy array so app.py can load it easily
    np.save("labels.npy", np.array(labels)) 
    
    print("-" * 30)
    print(f"✅ Success! Created labels.npy with: {labels}")
    print(f"✅ Dataset ready: X={X.shape}, y={y.shape}")
    print("-" * 30)
else:
    print("❌ ERROR: No .npy files found in data/raw folders. Check your paths!")