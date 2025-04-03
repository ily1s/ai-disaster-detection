import h5py
try:
    with h5py.File('/Users/ilyas/myenv/YARBI/AI4SDG/models/model_v2.weights.h5', 'r') as f:
        print("File loaded successfully!")
except OSError as e:
    print(f"Error: {e}")
