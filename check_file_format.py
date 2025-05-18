import pickle
import joblib
import numpy as np
import os

def try_pickle_load():
    try:
        with open('digits_model.pkl', 'rb') as file:
            model = pickle.load(file)
        print("Successfully loaded with pickle!")
        return model
    except Exception as e:
        print(f"Pickle load failed: {str(e)}")
        return None

def try_joblib_load():
    try:
        model = joblib.load('digits_model.pkl')
        print("Successfully loaded with joblib!")
        return model
    except Exception as e:
        print(f"Joblib load failed: {str(e)}")
        return None

def examine_file():
    try:
        # Check file size and first few bytes
        file_size = os.path.getsize('digits_model.pkl')
        print(f"\nFile size: {file_size} bytes")
        
        with open('digits_model.pkl', 'rb') as file:
            header = file.read(20)  # Read first 20 bytes
            print("\nFirst 20 bytes (hex):")
            print(' '.join(f'{b:02x}' for b in header))
            
    except Exception as e:
        print(f"Error examining file: {str(e)}")

if __name__ == "__main__":
    print("Attempting to load the model in different ways...")
    print("\n1. Trying pickle.load():")
    model = try_pickle_load()
    
    if model is None:
        print("\n2. Trying joblib.load():")
        model = try_joblib_load()
    
    print("\nExamining file format:")
    examine_file() 